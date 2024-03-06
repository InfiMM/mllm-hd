import torch
import torch.nn as nn

from functools import partial
from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from transformers import AutoModelForCausalLM, AutoTokenizer
from .eva_vit import CLIPVisionCfg, EVAVisionTransformer
from .utils import _infer_decoder_layers_attr_name, extend_instance


class InfiMMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_config = config['visual']
        vision_encoder = self.build_vision_encoder()
        self.language_config = config['language']
        self.tokenizer = self.build_tokenizer()
        language_encoder = self.build_language_encoder()
        self.model = self.build_flamingo(vision_encoder, language_encoder)

        self.model.requires_grad_(False)
        if not self.config['freeze_vit']:
            self.model.vision_encoder.requires_grad_(True)
            self.model.vit_use_grad = True
        if not self.config['freeze_llm']:
            self.model.lang_encoder.requires_grad_(True)
        if not self.config['freeze_lm_embeddings']:
            self.model.lang_encoder.get_input_embeddings().requires_grad_(True)

        self.model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        

        self.media_token_id = self.config['image_token_id']
        self.endofchunk_token_id = self.config['eoc_token_id']

    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.language_config["_name_or_path"])
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    '<|endofchunk|>',
                    '<image>',
                ]
            }
        )

        self.pad_already_exist = True
        if tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            self.pad_already_exist = False

        return tokenizer

    def build_vision_encoder(self):
        vision_cfg = CLIPVisionCfg(**self.vision_config)
        vision_encoder = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=vision_cfg.embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool,  # False
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_cfg.width // vision_cfg.head_width,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len=vision_cfg.pt_hw_seq_len,  # 224/14
            intp_freq=vision_cfg.intp_freq,
            naiveswiglu=vision_cfg.naiveswiglu,
            subln=vision_cfg.subln,
        )

        return vision_encoder

    def build_language_encoder(self):
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            self.language_config["_name_or_path"]
        )
        lang_encoder.resize_token_embeddings(self.language_config["vocab_size"])
        return lang_encoder

    def build_flamingo(self, vision_encoder, lang_encoder):
        extend_instance(lang_encoder, FlamingoLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        model = Flamingo(
            vision_encoder,
            lang_encoder,
            self.config['eoc_token_id'],
            self.config['image_token_id'],
            vis_dim=self.vision_config["width"],
            cross_attn_every_n_layers=self.config['cross_attn_every_n_layers'],
            gradient_checkpointing=self.config['use_grad_checkpoint'],
        )
        checkpoints = torch.load(self.config['pretrained_model_path'], map_location="cpu")
        states = model.load_state_dict(checkpoints)
        print(states)

        return model

    def forward(self, samples):
        # [bs, T_img, F, C, H ,W], T_img is the num of images, while F means frame num. (in this case, we do not apply videos, so please set it to 1.)
        images = samples["image"].contiguous() 
        # [bs, N], N is the sequence length
        input_ids = samples["input_ids"].contiguous()
        # [bs, N] this is the attention mask for text sequence
        attention_mask = samples["attention_mask"].contiguous()
        # [bs, N] this is the mask for label
        question_mask = samples["question_mask"].contiguous()

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[question_mask == 1] = -100
        labels[labels == self.config['image_token_id']] = -100
        loss = self.model(
            vision_x=images,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

        return loss


    def generate(
        self,
        batch_images,
        input_ids,
        attention_mask,
        **kwargs,
    ):
        
        with torch.inference_mode():
            outputs = self.model.generate(
                batch_images,
                input_ids,
                attention_mask,
                **kwargs,
            )

        # Extract only the new gnerated tokens
        outputs = outputs[:, len(input_ids[0]) :]
        self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs
