"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import torch
from einops import rearrange, repeat
from torch import einsum, nn

from einops_exts import rearrange_many

try:
    from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
except:
    from torch.utils.checkpoint import checkpoint


def exists(val):
    return val is not None


def FeedForward(
    dim,
    mult=4,
    enable_init_network_params=False,
    initializer_range=0.02,
):
    inner_dim = int(dim * mult)
    net = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

    if enable_init_network_params:
        # then start the initialization
        net[0].weight.data.normal_(mean=0.0, std=initializer_range)
        net[0].bias.data.zero_()
        net[1].weight.data.normal_(mean=0.0, std=initializer_range)
        net[3].weight.data.normal_(mean=0.0, std=initializer_range)
    return net


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        enable_init_network_params=False,
        initializer_range=0.02,
    ):
        super().__init__()

        self.scale = dim_head**-0.5
        self.heads = heads
        self.initializer_range = initializer_range

        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        if enable_init_network_params:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents.contiguous())

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
        enable_init_network_params=False,
        initializer_range=0.02,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing
        self.initializer_range = initializer_range

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            enable_init_network_params=enable_init_network_params,
                            initializer_range=initializer_range,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            enable_init_network_params=enable_init_network_params,
                            initializer_range=initializer_range,
                        ),
                    ]
                )
            )
        # Should this norm layer also change?
        self.norm = nn.LayerNorm(dim)
        if enable_init_network_params:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.initializer_range)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """

        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            if self.gradient_checkpointing and latents.requires_grad:
                latents = checkpoint(attn, x, (latents)) + latents
                latents = checkpoint(ff, latents) + latents
            else:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents

        return self.norm(latents)


# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
        enable_init_network_params=False,
        initializer_range=0.02,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.initializer_range = initializer_range
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

        if enable_init_network_params:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """

        if not use_cached_media:
            assert media_locations.shape[1] == x.shape[1], (
                f"media_location.shape is {media_locations.shape} but x.shape is"
                f" {x.shape}"
            )

        T_txt = x.shape[1]
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x.contiguous())
        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)

        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1

            if use_cached_media:
                # text time is set to the last cached media location
                text_time = repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    "b -> b i",
                    i=T_txt,
                )
            else:
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1)

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )

            if self.only_attend_immediate_media:
                # any text without a preceding media needs to have attention zeroed out
                text_without_media_mask = text_time == 0
                text_without_media_mask = rearrange(
                    text_without_media_mask, "b i -> b 1 i 1"
                )

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale
        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=12,
        ff_mult=1,
        only_attend_immediate_media=True,
        enable_init_network_params=False,
        initializer_range=0.02,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
            enable_init_network_params=enable_init_network_params,
            initializer_range=initializer_range,
        )
        self.attn_gate = nn.Parameter(torch.zeros(dim))
        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.zeros(dim))
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        x,
        media,
        media_locations=None,
        use_cached_media=False,
    ):
        if exists(media_locations):
            flag = torch.sum(media_locations, dim=-1)
            flag = torch.where(flag > 0.0, 1.0, 0.0)
            flag = flag.unsqueeze(1).unsqueeze(1).to(torch.bfloat16)
        else:
            flag = 1.0

        if self.gradient_checkpointing and media.requires_grad:
            x = (
                flag
                * checkpoint(self.attn, x, media, media_locations, use_cached_media)
                * self.attn_gate.tanh()
                + x
            )
            x = flag * checkpoint(self.ff, x) * self.ff_gate.tanh() + x

        else:
            x = (
                flag
                * self.attn(
                    x,
                    media,
                    media_locations=media_locations,
                    use_cached_media=use_cached_media,
                )
                * self.attn_gate.tanh()
                + x
            )
            x = flag * self.ff(x) * self.ff_gate.tanh() + x

        return x