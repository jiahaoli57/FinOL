import torch
import torch.nn.functional as F

from functools import wraps
from torch import nn, einsum
from einops import rearrange, repeat
from finol.config import *

MODEL_CONFIG = MODEL_CONFIG.get("LSRE-CAAN", {})
NUM_LAYERS = MODEL_CONFIG["NUM_LAYERS"]
NUM_LATENTS = MODEL_CONFIG["NUM_LATENTS"]
LATENT_DIM = MODEL_CONFIG["LATENT_DIM"]
CROSS_HEADS = MODEL_CONFIG["CROSS_HEADS"]
LATENT_HEADS = MODEL_CONFIG["LATENT_HEADS"]
CROSS_DIM_HEAD = MODEL_CONFIG["CROSS_DIM_HEAD"]
LATENT_DIM_HEAD = MODEL_CONFIG["LATENT_DIM_HEAD"]
DROPOUT = MODEL_CONFIG["DROPOUT"]

"""
                       Table 7: Hyper-parameters of the LSRE-CAAN framework.
+---------------------------+--------+------------------------------------------------------------------+
| Hyper-parameter           | Choice | Description                                                      |
+---------------------------+--------+------------------------------------------------------------------+
| Depth of net (L)          | 1      | The number of process layers in LSRE.                            |
| Number of latents (M)     | 1      | The number of latents.                                           |
| Latent dimension (D)      | 32     | The size of the latent space.                                    | 
| Number of cross-heads     | 1      | The number of heads for cross-attention.                         | 
| Number of latent-heads    | 1      | The number of heads for latent self-attention.                   | 
| Cross-attention dimension | 64     | The number of dimensions per cross-attention head.               | 
| Self-attention dimension  | 32     | The number of dimensions per latent self-attention head.         | 
| Dropout ratio             | None   | No dropout is used following Jaegle et al. (2022).               | 
| Embedding dimension       | None   | No Embedding layer is used, as illustrated in Section 4.1.       | 
+---------------------------+--------+------------------------------------------------------------------+
| Optimizer                 | LAMB   | An optimizer specifically designed for Transformer-based models. |
| Learning rate             | 0.001  | Parameter of the LAMB optimizer.                                 |
| Weight decay rate         | 0.01   | Parameter of the LAMB optimizer.                                 |
| Training steps            | 10^4   | Training times.                                                  |
| Episode length (T)        | 50     | The length of an episode.                                        |
+---------------------------+--------+------------------------------------------------------------------+
| G                         | m/2    | Half of the assets are identified as winners.                    |
| W                         | 100    | The look-back window size.                                       |
+---------------------------+--------+------------------------------------------------------------------+
"""


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None, device=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class QuickGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            # QuickGELU(),
            # nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)  # query_dim = latent_dim

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)  # return context if exists(context) else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class LSRE(nn.Module):
    r"""
    This class implements the Long Sequence Representations Extractor (LSRE) module

    For more details, please refer to the papers `Online portfolio management via deep reinforcement learning with
    high-frequency data <https://www.sciencedirect.com/science/article/abs/pii/S030645732200348X>` and `Perceiver IO: A
    General Architecture for Structured Inputs & Outputs <https://arxiv.org/abs/2107.14795>`
    """

    def __init__(
            self,
            *,
            depth,
            dim,
            num_latents,
            latent_dim,
            cross_heads,
            latent_heads,
            cross_dim_head,
            latent_dim_head,
            weight_tie_layers=True,
            device,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head, device=DEVICE),
                    context_dim=dim, device=DEVICE),
            PreNorm(latent_dim, FeedForward(latent_dim, device=device), device=DEVICE)
        ])
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    device=DEVICE), device=DEVICE)
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, device=DEVICE), device=DEVICE)
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(
            self,
            data,
            mask=None,
            queries=None
    ):
        # b, *_, device = *data.shape, data.device
        b, *_ = *data.shape, data.device

        # latents
        x = repeat(self.latents, 'n d -> b n d', b=b)
        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        # layers
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # return torch.mean(x, dim=1)  # [batch_size * NUM_ASSETS, num_latents, latent_dim] -> [batch_size * NUM_ASSETS, latent_dim]
        return x[:, -1, :].squeeze()  # [batch_size * NUM_ASSETS, num_latents, latent_dim] -> [batch_size * NUM_ASSETS, latent_dim]


class LSRE_CAAN(nn.Module):
    r"""
    This class implements the LSRE_CAAN model

    For more details, please refer to the paper `Online portfolio management via deep reinforcement learning with
    high-frequency data <https://www.sciencedirect.com/science/article/abs/pii/S030645732200348X>`
    """

    def __init__(
            self,
            *,
            num_assets,
            num_features_augmented,
            num_features_original,
            window_size,
            **kwargs
    ):
        super().__init__()
        self.num_assets = num_assets
        self.num_features_augmented = num_features_augmented
        self.num_features_original = num_features_original
        self.window_size = window_size
        self.dim = num_features_original
        # self.token_emb = nn.Linear(num_features_original, self.dim)
        self.pos_emb = nn.Embedding(window_size, self.dim)
        self.latent_dim = LATENT_DIM

        self.lsre = LSRE(
            depth=NUM_LAYERS,
            dim=self.dim,  # num_feats
            num_latents=NUM_LATENTS,
            # latent_dim=LATENT_DIM,
            latent_dim=self.latent_dim,
            cross_heads=CROSS_HEADS,
            latent_heads=LATENT_HEADS,
            cross_dim_head=CROSS_DIM_HEAD,
            latent_dim_head=LATENT_DIM_HEAD,
            device=DEVICE,
            **kwargs
        )

        value_dim = LATENT_DIM
        self.linear_query = torch.nn.Linear(value_dim, value_dim)
        self.linear_key = torch.nn.Linear(value_dim, value_dim)
        self.linear_value = torch.nn.Linear(value_dim, value_dim)
        self.linear_winner = torch.nn.Linear(value_dim, 1)
        self.dropout = nn.Dropout(p=DROPOUT)
        self.lsre_linear = torch.nn.Linear(LATENT_DIM, 1)

        if MODEL_NAME == 'LSRE-CAAN-d':
            self.ab_study_linear_1 = torch.nn.Linear(num_features_original, value_dim)
        if MODEL_NAME == 'LSRE-CAAN-dd':
            self.ab_study_linear_1 = torch.nn.Linear(value_dim, 1)

    def forward(
            self,
            x
    ):
        # Input Transformation
        batch_size, num_assets, num_features_augmented = x.shape  # n: window size; d: number of features
        window_size = self.window_size
        num_features_original = self.num_features_original

        x = x.view(batch_size, num_assets, window_size, num_features_original)
        x = rearrange(x, 'b m n d -> (b m) n d')

        n, d = x.shape[1], x.shape[2]  # n: window size; d: number of features

        # LSRE
        # x = self.token_emb(x)  # optional
        pos_emb = self.pos_emb(torch.arange(n, device=DEVICE))

        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        if MODEL_NAME == 'LSRE-CAAN-d':
            # stock_rep = torch.mean(x, dim=1)
            stock_rep = rearrange(x, 'b n d -> b d n')
            stock_rep = stock_rep[:, :, -1].squeeze(-1)
            stock_rep = self.ab_study_linear_1(stock_rep)
        else:
            stock_rep = self.lsre(x, mask=None, queries=None)  # [batch_size * num_assets, LATENT_DIM]

        stock_rep = self.dropout(stock_rep)
        x = stock_rep.view(batch_size, num_assets, self.latent_dim)

        if MODEL_NAME == 'LSRE-CAAN-dd':
            final_scores = self.ab_study_linear_1(x).squeeze(-1)
        else:
            # CAAN
            query = self.linear_query(x)  # [batch_size, num_assets, LATENT_DIM]
            key = self.linear_key(x)  # [batch_size, num_assets, LATENT_DIM]
            value = self.linear_value(x)  # [batch_size, num_assets, LATENT_DIM]

            beta = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(float(query.shape[-1])))  # [batch_size, num_assets, LATENT_DIM]
            beta = F.softmax(beta, dim=-1).unsqueeze(-1)
            x = torch.sum(value.unsqueeze(1) * beta, dim=2)  # [batch_size, num_assets, LATENT_DIM]

            final_scores = self.linear_winner(x).squeeze(-1)  # [batch_size, num_assets]

        return final_scores

