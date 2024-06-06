import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat
from finol.config import *

MODEL_CONFIG = MODEL_CONFIG.get("AlphaPortfolio", {})
DIM_EMBEDDING = MODEL_CONFIG["DIM_EMBEDDING"]  # 256
DIM_FEEDFORWARD = MODEL_CONFIG["DIM_FEEDFORWARD"]  # 1024
NUM_HEADS = MODEL_CONFIG["NUM_HEADS"]  # 4
NUM_LAYERS = MODEL_CONFIG["NUM_LAYERS"]  # 1
DROPOUT = MODEL_CONFIG["DROPOUT"]  # 0.2

"""
     Table C.1: Hyperparameters of TE-CAAN-Based AP
+----------------------+--------+-----------------+--------+
| Hyper-parameter      | Choice | Hyper-parameter | Choice |
+----------------------+--------+-----------------+--------+
| Embedding dimension  | 256    | Optimizer       | SGD    |
| Feed-forward network | 1021   | Learning rate   | 0.0001 |
| Number of multi-head | 4      | Dropout ratio   | 0.2    |
| Number of TE layer   | 1      | Training epochs | 30     |
+----------------------+--------+-----------------+--------+
"""


class SREM(nn.Module):
    r"""
    This class implements the Sequence Representations Extraction (SREM) module

    For more details, please refer to the papers `AlphaPortfolio: Direct Construction through Reinforcement Learning
    and Interpretable AI <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3698800>` and `Attention is all you need
    <https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>`
    """
    def __init__(self, num_features_original, window_size):
        super().__init__()
        self.token_emb = nn.Linear(num_features_original, DIM_EMBEDDING)
        self.pos_emb = nn.Embedding(window_size, DIM_EMBEDDING)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=DIM_EMBEDDING,
                nhead=NUM_HEADS,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout=DROPOUT,
                batch_first=True,
            ),
            num_layers=NUM_LAYERS,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): the sequence to the encoder (required).
                        shape (batch_size * num_assets, window_size, num_features_original)
        """
        n, d = x.shape[1], x.shape[2]  # n: window size; d: number of features
        x = self.token_emb(x)  # [batch_size * num_assets, window_size, num_features_original] -> [batch_size * num_assets, window_size, DIM_EMBEDDING]
        pos_emb = self.pos_emb(torch.arange(n, device=DEVICE))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        x = self.transformer_encoder(x)  # [batch_size * num_assets, window_size, DIM_EMBEDDING] -> [batch_size * num_assets, window_size, DIM_EMBEDDING]

        return torch.mean(x, dim=1)  # [batch_size * num_assets, window_size, DIM_EMBEDDING] -> [batch_size * num_assets, DIM_EMBEDDING]


class CAAN(nn.Module):
    r"""
    This class implements the Cross Asset Attention Network (CAAN) module
    """
    def __init__(self, DIM_EMBEDDING):
        super().__init__()
        self.linear_query = torch.nn.Linear(DIM_EMBEDDING, DIM_EMBEDDING)
        self.linear_key = torch.nn.Linear(DIM_EMBEDDING, DIM_EMBEDDING)
        self.linear_value = torch.nn.Linear(DIM_EMBEDDING, DIM_EMBEDDING)
        self.linear_winner = torch.nn.Linear(DIM_EMBEDDING, 1)

    def forward(self, x):
        query = self.linear_query(x)  # [batch_size, num_assets, DIM_EMBEDDING]
        key = self.linear_key(x)  # [batch_size, num_assets, DIM_EMBEDDING]
        value = self.linear_value(x)  # [batch_size, num_assets, DIM_EMBEDDING]

        beta = torch.matmul(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(float(query.shape[-1])))  # [batch_size, num_assets, DIM_EMBEDDING]
        beta = F.softmax(beta, dim=-1).unsqueeze(-1)
        x = torch.sum(value.unsqueeze(1) * beta, dim=2)  # [batch_size, num_assets, DIM_EMBEDDING]

        final_scores = self.linear_winner(x).squeeze(-1)  # [batch_size, num_assets]

        return final_scores


class AlphaPortfolio(nn.Module):
    r"""
    This class implements the AlphaPortfolio model

    For more details, please refer to the paper `AlphaPortfolio: Direct Construction through Reinforcement Learning
    and Interpretable AI <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3698800>`
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
        self.num_features_original = num_features_original
        self.window_size = window_size
        self.srem = SREM(num_features_original, window_size)
        self.caan = CAAN(DIM_EMBEDDING)

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

        # Sequence Representations Extraction (SREM)
        stock_rep = self.srem(x)  # [batch_size * num_assets, window_size, DIM_EMBEDDING] -> [batch_size * num_assets, DIM_EMBEDDING]

        x = stock_rep.view(batch_size, num_assets, DIM_EMBEDDING)  # [batch_size * num_assets, DIM_EMBEDDING] -> [batch_size, num_assets, DIM_EMBEDDING]

        # Cross Asset Attention Network (CAAN)
        final_scores = self.caan(x)

        return final_scores


if __name__ == '__main__':
    torch.manual_seed(MANUAL_SEED)
    batch_size = 128
    num_assets = 6
    window_size = 30
    num_features_original = 10
    num_features_augmented = window_size * num_features_original
    x = torch.ones(batch_size, num_assets, num_features_augmented).to(DEVICE)
    model = AlphaPortfolio(
        num_assets=num_assets,
        num_features_augmented=num_features_augmented,
        num_features_original=num_features_original,
        window_size=window_size,
    ).to(DEVICE)
    final_scores = model(x)
    print(final_scores)

