# Pretrain model
Just download the model at https://huggingface.co/lllliuhhhhggg/infimm_pretrain/tree/main. We provide two pretraining models in our paper (only stage1) https://arxiv.org/abs/2403.01487. It is a Flamingo style model, the only difference is we remove the perceiver resampler. We use vit-e and vicuna in our model.
These models are pretrained on mmc4, obelisc, coyo238m (sampled from coyo700m), laion115m, laioncoco. Our model's training speed is much faster than LLaVA due to the cross attention information fusion (using same amount of data). Feel free to build something from our pretrained model. 
# Training
As we utilize the company's training framework, we can not provide the training code directly. So here we give you a demo of the data process and forward pass in demo_forward.py

# Citation

```latex
@misc{liu2024infimmhd,
      title={InfiMM-HD: A Leap Forward in High-Resolution Multimodal Understanding}, 
      author={Haogeng Liu and Quanzeng You and Xiaotian Han and Yiqi Wang and Bohan Zhai and Yongfei Liu and Yunzhe Tao and Huaibo Huang and Ran He and Hongxia Yang},
      year={2024},
      eprint={2403.01487},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
