import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from finol.data_layer.ScalerSelector import ScalerSelector
from finol.utils import load_config

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
    """
    This class implements the Sequence Representations Extraction (SREM) module

    For more details, please refer to the papers `AlphaPortfolio: Direct Construction through Reinforcement Learning
    and Interpretable AI <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3698800>` and `Attention is all you need
    <https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>`
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.token_emb = nn.Linear(model_args["NUM_FEATURES_ORIGINAL"], model_params["DIM_EMBEDDING"])
        self.pos_emb = nn.Embedding(model_args["WINDOW_SIZE"], model_params["DIM_EMBEDDING"])
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_params["DIM_EMBEDDING"],
                nhead=model_params["NUM_HEADS"],
                dim_feedforward=model_params["DIM_FEEDFORWARD"],
                dropout=model_params["DROPOUT"],
                batch_first=True,
            ),
            num_layers=model_params["NUM_LAYERS"],
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): the sequence to the encoder (required).
                        shape (batch_size * num_assets, window_size, num_features_original)
        """
        _, n, d, device = x.shape[0], x.shape[1], x.shape[2], x.device  # n: window size; d: number of features
        x = self.token_emb(x)  # [batch_size * num_assets, window_size, num_features_original] -> [batch_size * num_assets, window_size, DIM_EMBEDDING]
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, "n d -> () n d")
        x = x + pos_emb

        x = self.transformer_encoder(x)  # [batch_size * num_assets, window_size, DIM_EMBEDDING] -> [batch_size * num_assets, window_size, DIM_EMBEDDING]

        return torch.mean(x, dim=1)  # [batch_size * num_assets, window_size, DIM_EMBEDDING] -> [batch_size * num_assets, DIM_EMBEDDING]


class CAAN(nn.Module):
    """
    This class implements the Cross Asset Attention Network (CAAN) module
    """
    def __init__(self, model_params):
        super().__init__()
        self.linear_query = torch.nn.Linear(model_params["DIM_EMBEDDING"], model_params["DIM_EMBEDDING"])
        self.linear_key = torch.nn.Linear(model_params["DIM_EMBEDDING"], model_params["DIM_EMBEDDING"])
        self.linear_value = torch.nn.Linear(model_params["DIM_EMBEDDING"], model_params["DIM_EMBEDDING"])
        self.linear_winner = torch.nn.Linear(model_params["DIM_EMBEDDING"], 1)

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
    """
    This class implements the AlphaPortfolio model

    For more details, please refer to the paper `AlphaPortfolio: Direct Construction through Reinforcement Learning
    and Interpretable AI <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3698800>`
    """
    def __init__(self, model_args, model_params):
        super().__init__()
        self.config = load_config()
        self.model_args = model_args
        self.model_params = model_params

        self.srem = SREM(model_args, model_params)
        self.caan = CAAN(model_params)

    def forward(self, x):
        batch_size, num_assets, num_features_augmented = x.shape

        """Input Transformation"""
        x = x.view(batch_size, num_assets, self.model_args["WINDOW_SIZE"], self.model_args["NUM_FEATURES_ORIGINAL"])
        x = rearrange(x, "b m n d -> (b m) n d")
        if self.config["SCALER"].startswith("Window"):
            x = ScalerSelector().window_normalize(x)

        """Sequence Representations Extraction (SREM)"""
        stock_rep = self.srem(x)  # [batch_size * num_assets, window_size, DIM_EMBEDDING] -> [batch_size * num_assets, DIM_EMBEDDING]
        x = stock_rep.view(batch_size, num_assets, self.model_params["DIM_EMBEDDING"])  # [batch_size * num_assets, DIM_EMBEDDING] -> [batch_size, num_assets, DIM_EMBEDDING]

        """Cross Asset Attention Network (CAAN)"""
        final_scores = self.caan(x)

        return final_scores


if __name__ == "__main__":
    # config = load_config()
    DEVICE = "cuda"
    torch.manual_seed(0)
    batch_size = 128
    num_assets = 6
    window_size = 30
    num_features_original = 10
    num_features_augmented = window_size * num_features_original
    # x = torch.ones(batch_size, num_assets, num_features_augmented).to(DEVICE)
    x = torch.rand(batch_size, num_assets, num_features_augmented).to(DEVICE)
    model_args = {
        "NUM_FEATURES_ORIGINAL": num_features_original,
        "WINDOW_SIZE": window_size,
    }
    model_params = {
        "DIM_EMBEDDING": 256,
        "DIM_FEEDFORWARD": 1021,
        "NUM_HEADS": 4,
        "NUM_LAYERS": 1,
        "DROPOUT": 0.2,
    }
    model = AlphaPortfolio(model_args, model_params).to(DEVICE)
    final_scores = model(x)
    print(final_scores)

