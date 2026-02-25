import functools
from typing import Tuple
import torch
import torch.nn as nn
import math

from functools import lru_cache
from torch.nn.functional import scaled_dot_product_attention


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Embed(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels,):
        embeddings = self.embedding_table(labels)
        return embeddings

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        x =  self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x

def precompute_freqs_cis_2d(dim: int, height: int, width:int, theta: float = 10000.0, scale=16.0):
    # assert  H * H == end
    # flat_patch_pos = torch.linspace(-1, 1, end) # N = end
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B N H Hc
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # B, H, N, Hc
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()  # B, H, N, Hc
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4.0, ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x,  c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels+max_freqs**2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        pos_x = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        pos_y, pos_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)

        freqs = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        freqs_x = freqs[None, :, None]
        freqs_y = freqs[None, None, :]
        coeffs = (1 + freqs_x * freqs_y) ** -1
        dct_x = torch.cos(pos_x * freqs_x * torch.pi)
        dct_y = torch.cos(pos_y * freqs_y * torch.pi)
        dct = (dct_x * dct_y * coeffs).view(1, -1, self.max_freqs ** 2)
        return dct


    def forward(self, inputs):
        B, P2, C = inputs.shape
        patch_size = int(P2 ** 0.5)
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs


class SharedPointEncoder(nn.Module):
    def __init__(self, hidden_size_x, mlp_ratio=2):
        super().__init__()
        self.norm = RMSNorm(hidden_size_x, eps=1e-6)
        hidden = int(hidden_size_x * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size_x, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden_size_x, bias=True)

    def forward(self, x):
        res_x = x
        x = self.norm(x)
        x = self.fc1(x)
        x = torch.nn.functional.silu(x)
        x = self.fc2(x)
        return x + res_x


class PatchConditionHead(nn.Module):
    def __init__(self, hidden_size_s, condition_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size_s, condition_size, bias=True),
            nn.SiLU(),
            nn.Linear(condition_size, condition_size, bias=True),
        )

    def forward(self, s):
        return self.net(s)


class ConditionedPointEncoder(nn.Module):
    def __init__(self, hidden_size_x, condition_size, mlp_ratio=2):
        super().__init__()
        self.norm = RMSNorm(hidden_size_x, eps=1e-6)
        hidden = int(hidden_size_x * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size_x, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden_size_x, bias=True)
        self.cond_proj = nn.Linear(condition_size, 3 * hidden_size_x, bias=True)

    def forward(self, x, condition):
        shift, scale, gate = self.cond_proj(condition).chunk(3, dim=-1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        gate = gate[:, None, :]
        res_x = x
        x = modulate(self.norm(x), shift, scale)
        x = self.fc1(x)
        x = torch.nn.functional.silu(x)
        x = self.fc2(x)
        return res_x + gate * x


class ConditionedQueryHead(nn.Module):
    def __init__(self, hidden_size_x, out_channels, condition_size):
        super().__init__()
        self.norm = RMSNorm(hidden_size_x, eps=1e-6)
        self.cond_proj = nn.Linear(condition_size, 2 * hidden_size_x, bias=True)
        self.linear = nn.Linear(hidden_size_x, out_channels, bias=True)

    def forward(self, x, condition):
        shift, scale = self.cond_proj(condition).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift[:, None, :], scale[:, None, :])
        return self.linear(x)

class PixNerDiT(nn.Module):
    def __init__(
            self,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            hidden_size_x=64,
            nerf_mlpratio=4,
            num_blocks=18,
            num_cond_blocks=4,
            patch_size=2,
            num_classes=1000,
            learn_sigma=True,
            deep_supervision=0,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.num_cond_blocks = num_cond_blocks
        self.num_decoder_blocks = self.num_blocks - self.num_cond_blocks
        self.patch_size = patch_size
        self.x_embedder = NerfEmbedder(in_channels, hidden_size_x, max_freqs=8)
        self.s_embedder = Embed(in_channels*patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes+1, hidden_size)
        self.condition_size = hidden_size_x
        self.patch_condition_head = PatchConditionHead(self.hidden_size, self.condition_size)
        self.shared_point_encoder = SharedPointEncoder(hidden_size_x, nerf_mlpratio)
        self.decoder_blocks = nn.ModuleList([
            ConditionedPointEncoder(hidden_size_x, self.condition_size, nerf_mlpratio)
            for _ in range(self.num_decoder_blocks)
        ])
        self.final_layer = ConditionedQueryHead(hidden_size_x, self.out_channels, self.condition_size)

        self.weight_path = weight_path

        self.load_ema = load_ema
        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_cond_blocks)
        ])
        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero init final layer
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)


    def forward(self, x, t, y, s=None, mask=None):
        B, _, H, W = x.shape
        pos = self.fetch_pos(H//self.patch_size, W//self.patch_size, x.device)
        # 把图像拆成了patch token
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(B, 1, self.hidden_size)
        c = nn.functional.silu(t + y)
        if s is None:
            s = self.s_embedder(x)
            for i in range(self.num_cond_blocks):
                s = self.blocks[i](s, c, pos, mask)
            s = nn.functional.silu(t + s)
        batch_size, length, _ = s.shape
        x = x.reshape(batch_size*length, self.in_channels, self.patch_size**2)
        x = x.transpose(1, 2)
        s = s.view(batch_size*length, self.hidden_size)
        condition = self.patch_condition_head(s)
        x = self.x_embedder(x)
        x = self.shared_point_encoder(x)
        for block in self.decoder_blocks:
            x = block(x, condition)
        x = self.final_layer(x, condition)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, length, -1)
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x
