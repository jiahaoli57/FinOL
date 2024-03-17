import torch
import torch.nn.functional as F
from functools import wraps
from torch import nn, einsum
from einops import rearrange, repeat
from finol.config import *

NUM_LATENTS = MODEL_CONFIG.get("LSRE-CAAN")["NUM_LATENTS"]
LATENT_DIM = MODEL_CONFIG.get("LSRE-CAAN")["LATENT_DIM"]
CROSS_HEADS = MODEL_CONFIG.get("LSRE-CAAN")["CROSS_HEADS"]
LATENT_HEADS = MODEL_CONFIG.get("LSRE-CAAN")["LATENT_HEADS"]
CROSS_DIM_HEAD = MODEL_CONFIG.get("LSRE-CAAN")["CROSS_DIM_HEAD"]
LATENT_DIM_HEAD = MODEL_CONFIG.get("LSRE-CAAN")["LATENT_DIM_HEAD"]


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
        self.fn = fn.to(DEVICE)
        self.norm = nn.LayerNorm(dim).to(DEVICE)
        self.norm_context = nn.LayerNorm(context_dim).to(DEVICE) if exists(context_dim) else None

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
            nn.Linear(dim, dim).to(DEVICE),
            QuickGELU(),
        ).to(DEVICE)

    def forward(self, x):
        return self.net(x).to(DEVICE)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False).to(DEVICE)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False).to(DEVICE)
        self.to_out = nn.Linear(inner_dim, query_dim).to(DEVICE)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x).to(DEVICE)
        context = default(context, x)  # return context if exists(context) else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h).to(DEVICE)
            sim.masked_fill_(~mask, max_neg_value).to(DEVICE)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1).to(DEVICE)

        out = einsum('b i j, b j d -> b i d', attn, v).to(DEVICE)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).to(DEVICE)
        return self.to_out(out).to(DEVICE)


class LSRE(nn.Module):
    r"""
    This class implements the LSRE model proposed in my paper

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
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim)).to(DEVICE)
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head, device=DEVICE),
                    context_dim=dim, device=DEVICE).to(DEVICE),
            PreNorm(latent_dim, FeedForward(latent_dim, device=device).to(DEVICE), device=DEVICE)
        ])
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    device=DEVICE).to(DEVICE), device=DEVICE)
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, device=DEVICE).to(device), device=DEVICE)
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
            x = self_ff(x) + x  # x.shape = torch.Size([num_assets, num_latents, latent_dim])

        return torch.mean(x, dim=1)  # [batch_size, num_latents, latent_dim] -> [batch_size, latent_dim]


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
        self.Prop_winners = 1
        # self.token_emb = nn.Linear(num_features_original, self.dim).to(DEVICE)
        self.pos_emb = nn.Embedding(window_size, self.dim).to(DEVICE)
        self.latent_dim = LATENT_DIM

        depth = 1
        self.lsre = LSRE(
            depth=depth,  # 1
            dim=self.dim,  # num_feats
            num_latents=NUM_LATENTS,  # 1
            # latent_dim=LATENT_DIM,  # 32
            latent_dim=self.latent_dim,  # 32
            cross_heads=CROSS_HEADS,  # 1
            latent_heads=LATENT_HEADS,  # 1
            cross_dim_head=CROSS_DIM_HEAD,  # 64
            latent_dim_head=LATENT_DIM_HEAD,  # 32
            device=DEVICE,
            **kwargs
        )

        value_dim = LATENT_DIM
        self.linear_query = torch.nn.Linear(value_dim, value_dim).to(DEVICE)
        self.linear_key = torch.nn.Linear(value_dim, value_dim).to(DEVICE)
        self.linear_value = torch.nn.Linear(value_dim, value_dim).to(DEVICE)
        self.linear_winner = torch.nn.Linear(value_dim, 1).to(DEVICE)
        self.dropout = nn.Dropout(p=DROPOUT)

        self.lsre_linear = torch.nn.Linear(LATENT_DIM, 1).to(DEVICE)

    def forward(
            self,
            x
    ):
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
        stock_rep = self.lsre(x, mask=None, queries=None)  # [batch_size * num_assets, latent_dim]
        stock_rep = self.dropout(stock_rep)

        # CAAN
        x = stock_rep.view(batch_size, num_assets, self.latent_dim)

        query = self.linear_query(x)  # [batch_size, num_assets, LATENT_DIM]
        key = self.linear_key(x)  # [batch_size, num_assets, LATENT_DIM]
        value = self.linear_value(x)  # [batch_size, num_assets, LATENT_DIM]

        beta = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(float(query.shape[-1])))  # [BATCH_SIZE, num_assets, num_assets]
        beta = F.softmax(beta, dim=-1).unsqueeze(-1)
        x = torch.sum(value.unsqueeze(1) * beta, dim=2)  # [batch_size, num_assets, latent_dim]

        final_scores = self.linear_winner(x).squeeze()  # [BATCH_SIZE, NUM_ASSETS]

        # Portfolio Management
        # if self.Prop_winners != 1:
        #     # Prop_winners: proportion of winners, i.e. G in Section 4.2
        #     num_winners = int(self.num_assets * self.Prop_winners)
        #     assert num_winners != 0 and num_winners <= self.num_assets
        #     rank = torch.argsort(final_scores)
        #     winners = set(rank.detach().cpu().numpy()[-num_winners:])  # <class 'set'>
        #     winners_mask = torch.Tensor([0 if i in winners else 1 for i in range(rank.shape[0])]).to(DEVICE)
        #     portfolio = F.softmax(final_scores - 1e9 * winners_mask, dim=0)
        # else:

        portfolio = F.softmax(final_scores, dim=-1)
        return portfolio
