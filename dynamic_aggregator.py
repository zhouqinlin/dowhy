import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pandas as pd
import numpy as np
from tqdm import tqdm

class AttentionAggregator(nn.Module):
    """
    Embedding対応版 Attention Aggregator
    """
    def __init__(self, cat_configs, cont_dim, hidden_dim=64):
        super().__init__()
        # カテゴリ変数ごとのEmbedding層
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) 
            for vocab_size, emb_dim in cat_configs
        ])
        total_emb_dim = sum([emb_dim for _, emb_dim in cat_configs])
        input_dim = total_emb_dim + cont_dim
        
        self.query_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.key_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.scale = 1.0 / (hidden_dim ** 0.5)

    def forward(self, x_cat, x_cont, source_values):
        emb_outputs = []
        if x_cat.size(1) > 0:
            for i, emb_layer in enumerate(self.embeddings):
                emb_outputs.append(emb_layer(x_cat[:, i]))
        
        # 数値特徴量があれば結合
        features_list = emb_outputs + ([x_cont] if x_cont.size(1) > 0 else [])
        context_features = torch.cat(features_list, dim=1)

        Q = self.query_net(context_features)
        K = self.key_net(context_features)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        mask = torch.eye(scores.size(0), device=scores.device).bool()
        scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        
        aggregated_values = torch.matmul(weights, source_values)
        return aggregated_values, weights

class CausalRegressionModel(nn.Module):
    def __init__(self, own_feature_dim):
        super().__init__()
        self.linear = nn.Linear(own_feature_dim + 1, 1)

    def forward(self, own_features, aggregated_value):
        combined = torch.cat([own_features, aggregated_value], dim=1)
        return self.linear(combined)

class DynamicCausalModel(nn.Module):
    def __init__(self, cat_configs, cont_dim, own_feature_dim, hidden_dim=32):
        super().__init__()
        self.aggregator = AttentionAggregator(cat_configs, cont_dim, hidden_dim)
        self.predictor = CausalRegressionModel(own_feature_dim)

    def model(self, x_cat, x_cont, source_values, own_features, y_target=None):
        pyro.module("dcm", self)
        if y_target is not None:
            y_target = y_target.squeeze(-1)
        
        sigma = pyro.sample("sigma", dist.LogNormal(0., 1.))

        with pyro.plate("data", x_cat.size(0)):
            agg_values, _ = self.aggregator(x_cat, x_cont, source_values)
            prediction_mean = self.predictor(own_features, agg_values)
            prediction_mean = prediction_mean.squeeze(-1)
            pyro.sample("obs", dist.Normal(prediction_mean, sigma), obs=y_target)
        return agg_values

    def guide(self, x_cat, x_cont, source_values, own_features, y_target=None):
        sigma_loc = pyro.param("sigma_loc", torch.tensor(0.0))
        pyro.sample("sigma", dist.Delta(torch.exp(sigma_loc)))

class DynamicAggregatorHandler:
    def __init__(self, df, cat_context_cols, cont_context_cols, value_col, own_feature_cols, target_col, embedding_dim=16):
        self.cat_context_cols = cat_context_cols
        self.cont_context_cols = cont_context_cols
        self.value_col = value_col
        self.own_feature_cols = own_feature_cols
        self.target_col = target_col
        
        self.cat_configs = []
        cat_data_list = []
        for col in cat_context_cols:
            if df[col].dtype.name != 'category':
                series = df[col].astype('category')
            else:
                series = df[col]
            codes = series.cat.codes.values.astype(np.int64)
            vocab_size = len(series.cat.categories)
            self.cat_configs.append((vocab_size, embedding_dim))
            cat_data_list.append(codes)
            
        if cat_data_list:
            self.X_cat = torch.tensor(np.stack(cat_data_list, axis=1), dtype=torch.long)
        else:
            self.X_cat = torch.empty((len(df), 0), dtype=torch.long)
            
        if cont_context_cols:
            self.X_cont = torch.tensor(df[cont_context_cols].values, dtype=torch.float32)
        else:
            self.X_cont = torch.empty((len(df), 0), dtype=torch.float32)

        self.X_value = torch.tensor(df[[value_col]].values, dtype=torch.float32)
        self.X_own = torch.tensor(df[own_feature_cols].values, dtype=torch.float32)
        self.Y_target = torch.tensor(df[[target_col]].values, dtype=torch.float32)
        
        self.pyro_model = DynamicCausalModel(
            cat_configs=self.cat_configs,
            cont_dim=len(cont_context_cols),
            own_feature_dim=len(own_feature_cols)
        )

    def train(self, num_iterations=1000, lr=0.01, batch_size=None):
        pyro.clear_param_store()
        optimizer = Adam({"lr": lr})
        svi = SVI(self.pyro_model.model, self.pyro_model.guide, optimizer, loss=Trace_ELBO())

        pbar = tqdm(range(num_iterations))
        for i in pbar:
            if batch_size is None:
                loss = svi.step(self.X_cat, self.X_cont, self.X_value, self.X_own, self.Y_target)
            else:
                idx = torch.randperm(self.X_cat.size(0))[:batch_size]
                loss = svi.step(
                    self.X_cat[idx], 
                    self.X_cont[idx], 
                    self.X_value[idx], 
                    self.X_own[idx], 
                    self.Y_target[idx]
                )
            if i % 100 == 0:
                pbar.set_description(f"Loss: {loss:.4f}")

    def append_dynamic_agg_feature(self, df):
        self.pyro_model.eval()
        with torch.no_grad():
            agg_values, weights = self.pyro_model.aggregator(self.X_cat, self.X_cont, self.X_value)
        new_df = df.copy()
        new_col_name = f"DYNAMIC_{self.value_col}"
        new_df[new_col_name] = agg_values.numpy()
        return new_df