import torch
import torch.nn.functional as F

from functools import wraps
from torch import nn, einsum
from einops import rearrange, repeat
from finol.data_layer.scaler_selector import ScalerSelector
from finol.utils import load_config


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
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class QuickGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            # QuickGELU(),
            # nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
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

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LSRE(nn.Module):
    r"""
    This class implements the Long Sequence Representations Extractor (LSRE) module.

    For more details, please refer to the papers `Online portfolio management via deep reinforcement learning with
    high-frequency data <https://www.sciencedirect.com/science/article/abs/pii/S030645732200348X>`__ and `Perceiver IO: A
    General Architecture for Structured Inputs & Outputs <https://arxiv.org/abs/2107.14795>`__.
    """

    def __init__(self, model_args, model_params):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(model_params["NUM_LATENTS"], model_params["LATENT_DIM"]))
        # self.latents = nn.Parameter(torch.zeros(model_params["NUM_LATENTS"], model_params["LATENT_DIM"]))
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(model_params["LATENT_DIM"], Attention(model_params["LATENT_DIM"], model_args["num_features_original"], heads=model_params["CROSS_HEADS"], dim_head=model_params["CROSS_DIM_HEAD"]), context_dim=model_args["num_features_original"]),
            PreNorm(model_params["LATENT_DIM"], FeedForward(model_params["LATENT_DIM"]))
        ])
        get_latent_attn = lambda: PreNorm(model_params["LATENT_DIM"], Attention(model_params["LATENT_DIM"], heads=model_params["LATENT_HEADS"], dim_head=model_params["LATENT_DIM_HEAD"]))
        get_latent_ff = lambda: PreNorm(model_params["LATENT_DIM"], FeedForward(model_params["LATENT_DIM"]))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": True}

        for i in range(model_params["NUM_LAYERS"]):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(self, data, mask=None, queries=None):
        # b, *_, device = *data.shape, data.device
        b, *_ = *data.shape, data.device

        # latents
        x = repeat(self.latents, "n d -> b n d", b=b)
        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        # layers
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # return torch.mean(x, dim=1)  # [batch_size * NUM_ASSETS, num_latents, latent_dim] -> [batch_size * NUM_ASSETS, latent_dim]
        return x[:, -1, :].squeeze()  # [batch_size * NUM_ASSETS, num_latents, latent_dim] -> [batch_size * NUM_ASSETS, latent_dim]


class CAAN(nn.Module):
    """
    This class implements the Cross Asset Attention Network (CAAN) module.
    """
    def __init__(self, model_params):
        super().__init__()
        self.linear_query = torch.nn.Linear(model_params["LATENT_DIM"], model_params["LATENT_DIM"])
        self.linear_key = torch.nn.Linear(model_params["LATENT_DIM"], model_params["LATENT_DIM"])
        self.linear_value = torch.nn.Linear(model_params["LATENT_DIM"], model_params["LATENT_DIM"])
        self.linear_winner = torch.nn.Linear(model_params["LATENT_DIM"], 1)

    def forward(self, x):
        query = self.linear_query(x)  # [batch_size, num_assets, LATENT_DIM]
        key = self.linear_key(x)  # [batch_size, num_assets, LATENT_DIM]
        value = self.linear_value(x)  # [batch_size, num_assets, LATENT_DIM]

        beta = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(float(query.shape[-1])))  # [batch_size, num_assets, LATENT_DIM]
        beta = F.softmax(beta, dim=-1).unsqueeze(-1)
        x = torch.sum(value.unsqueeze(1) * beta, dim=2)  # [batch_size, num_assets, LATENT_DIM]

        final_scores = self.linear_winner(x).squeeze(-1)  # [batch_size, num_assets]
        return final_scores


class LSRE_CAAN(nn.Module):
    r"""
    Class to generate predicted scores for the input assets based on the LSRE-CAAN model.

    The LSRE-CAAN model is a Transformer-based model for asset scoring and portfolio selection. It consists of two
    main components:

    1. Long Sequence Representations Extractor (LSRE): This module uses a Transformer-based architecture to extract asset
    temporal representation. In addition, LSRE introduces a small set of latent units on top of the original Transformer
    Encoder to form an attention bottleneck through which the input must pass, which not only effectively solves the
    original Transformer Encoder's quadratic complexity problem.

    2. Cross Asset Attention Network (CAAN): This module takes the sequence representations generated by the LSRE and
    applies cross-asset attention to produce the final asset scores.

    The LSRE-CAAN model takes an input tensor ``x`` of shape ``(batch_size, num_assets, num_features_augmented)``,
    where ``num_features_augmented`` represents the number of features (including any preprocessed or augmented
    features) for each asset. The final output of the LSRE-CAAN model is a tensor of shape ``(batch_size, num_assets)``, where each element
    represents the predicted score for the corresponding asset.

    For more details, please refer to the paper `Online portfolio management via deep reinforcement learning with high-frequency data
    <https://www.sciencedirect.com/science/article/abs/pii/S030645732200348X>`__.

    .. table:: Table 7: Hyper-parameters of the LSRE-CAAN framework.
        :class: ghost

        +---------------------------+---------------+------------------------------------------------------------------+
        | Hyper-parameter           | Choice        | Description                                                      |
        +===========================+===============+==================================================================+
        | Depth of net (L)          | 1             | The number of process layers in LSRE.                            |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Number of latents (M)     | 1             | The number of latents.                                           |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Latent dimension (D)      | 32            | The size of the latent space.                                    |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Number of cross-heads     | 1             | The number of heads for cross-attention.                         |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Number of latent-heads    | 1             | The number of heads for latent self-attention.                   |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Cross-attention dimension | 64            | The number of dimensions per cross-attention head.               |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Self-attention dimension  | 32            | The number of dimensions per latent self-attention head.         |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Dropout ratio             | None          | No dropout is used following Jaegle et al. (2022).               |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Embedding dimension       | None          | No Embedding layer is used, as illustrated in Section 4.1.       |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Optimizer                 | LAMB          | An optimizer specifically designed for Transformer-based models. |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Learning rate             | 0.001         | Parameter of the LAMB optimizer.                                 |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Weight decay rate         | 0.01          | Parameter of the LAMB optimizer.                                 |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Training steps            | 10\ :sup:`4`\ | Training times.                                                  |
        +---------------------------+---------------+------------------------------------------------------------------+
        | Episode length (T)        | 50            | The length of an episode.                                        |
        +---------------------------+---------------+------------------------------------------------------------------+
        | G                         | m/2           | Half of the assets are identified as winners.                    |
        +---------------------------+---------------+------------------------------------------------------------------+
        | W                         | 100           | The look-back window size.                                       |
        +---------------------------+---------------+------------------------------------------------------------------+

    :param model_args: Dictionary containing model arguments, such as the number of features.
    :param model_params: Dictionary containing model hyperparameters, such as the number of layers, the number of latents, and the dropout rate.

    Example:
        .. code:: python
        >>> from finol.data_layer.dataset_loader import DatasetLoader
        >>> from finol.model_layer.model_instantiator import ModelInstantiator
        >>> from finol.utils import load_config, update_config, portfolio_selection
        >>>
        >>> # Configuration
        >>> config = load_config()
        >>> config["MODEL_NAME"] = "LSRE_CAAN"
        >>> config["MODEL_PARAMS"]["LSRE_CAAN"]["NUM_LAYERS"] = 1
        >>> config["MODEL_PARAMS"]["LSRE_CAAN"]["NUM_LATENTS"] = 12
        >>> config["MODEL_PARAMS"]["LSRE_CAAN"]["LATENT_DIM"] = 64
        >>> ...
        >>> update_config(config)
        >>>
        >>> # Data Layer
        >>> load_dataset_output = DatasetLoader().load_dataset()
        >>>
        >>> # Model Layer & Optimization Layer
        >>> ...
        >>> model = ModelInstantiator(load_dataset_output).instantiate_model()
        >>> print(f"model: {model}")
        >>> ...
        >>> train_loader = load_dataset_output["train_loader"]
        >>> for i, data in enumerate(train_loader, 1):
        ...     x_data, label = data
        ...     final_scores = model(x_data.float())
        ...     portfolio = portfolio_selection(final_scores)
        ...     print(f"batch {i} input shape: {x_data.shape}")
        ...     print(f"batch {i} label shape: {label.shape}")
        ...     print(f"batch {i} output shape: {portfolio.shape}")
        ...     print("-"*50)

    \
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        # self.token_emb = nn.Linear(num_features_original, self.dim)
        self.pos_emb = nn.Embedding(model_args["window_size"], model_args["num_features_original"])
        self.lsre = LSRE(model_args, model_params)
        self.caan = CAAN(model_params)
        self.dropout = nn.Dropout(p=self.model_params["DROPOUT"])
        if self.config["MODEL_NAME"] == "LSRE-CAAN-d":
            self.ab_study_linear_1 = torch.nn.Linear(model_args["num_features_original"], self.model_params["LATENT_DIM"])
        if self.config["MODEL_NAME"] == "LSRE-CAAN-dd":
            self.ab_study_linear_2 = torch.nn.Linear(self.model_params["LATENT_DIM"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape ``(batch_size, num_assets, num_features_augmented)``.
        :return: Output tensor of shape ``(batch_size, num_assets)`` containing the predicted scores for each asset.
        """
        batch_size, num_assets, num_features_augmented = x.shape
        device = x.device

        """Input Transformation"""
        x = x.view(batch_size, num_assets, self.model_args["window_size"], self.model_args["num_features_original"])
        x = rearrange(x, "b m n d -> (b m) n d")
        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        """Long Sequence Representations Extractor (LSRE)"""
        # x = self.token_emb(x)  # optional
        pos_emb = self.pos_emb(torch.arange(self.model_args["window_size"], device=device))
        pos_emb = rearrange(pos_emb, "n d -> () n d")
        x = x + pos_emb

        if self.config["MODEL_NAME"] == "LSRE-CAAN-d":
            # stock_rep = torch.mean(x, dim=1)
            stock_rep = rearrange(x, "b n d -> b d n")
            stock_rep = stock_rep[:, :, -1].squeeze(-1)
            stock_rep = self.ab_study_linear_1(stock_rep)
        else:
            stock_rep = self.lsre(x, mask=None, queries=None)  # [batch_size * num_assets, LATENT_DIM]

        stock_rep = self.dropout(stock_rep)
        x = stock_rep.view(batch_size, num_assets, self.model_params["LATENT_DIM"])

        """Cross Asset Attention Network (CAAN)"""
        if self.config["MODEL_NAME"] == "LSRE-CAAN-dd":
            final_scores = self.ab_study_linear_2(x).squeeze(-1)
        else:
            final_scores = self.caan(x)

        return final_scores