import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn import Linear
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class GenericGNN:
    def __init__(self, hidden_channels=64, out_channels=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.model = None
        
        # 前処理オブジェクトを保存する辞書
        self.scalers = {} 
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        # マッピング辞書 (グループ名 -> (値 -> ID))
        self.group_mappings = {}
        
        # 設定保存用
        self.cont_cols = []
        self.cat_cols = []
        self.group_cols = {} # {'group_name': 'column_name'}
        
        # 再現性のためにシード固定
        torch.manual_seed(42)

    def _prepare_features(self, df, is_training=True):
        """特徴量行列 X を作成する"""
        feature_parts = []
        
        # 1. 連続値の処理
        if self.cont_cols:
            cont_data = df[self.cont_cols].values
            if is_training:
                self.scaler = StandardScaler()
                cont_data = self.scaler.fit_transform(cont_data)
            else:
                cont_data = self.scaler.transform(cont_data)
            feature_parts.append(cont_data)
            
        # 2. カテゴリカル値の処理
        if self.cat_cols:
            cat_data = df[self.cat_cols]
            if is_training:
                cat_data = self.encoder.fit_transform(cat_data)
            else:
                cat_data = self.encoder.transform(cat_data)
            feature_parts.append(cat_data)
            
        if not feature_parts:
            raise ValueError("No features provided! Define continuous or categorical columns.")
            
        x = np.hstack(feature_parts)
        return torch.tensor(x, dtype=torch.float)

    def _build_graph(self, df, is_training=True):
        # 特徴量の準備
        x = self._prepare_features(df, is_training)
        N = len(df)
        
        # エッジの構築
        src = []
        dst = []
        
        # 仮想ノードの開始インデックス
        current_offset = N
        total_virtual_nodes = 0
        
        # 各グループ定義に基づいてエッジを作成
        for group_name, col_name in self.group_cols.items():
            if is_training:
                # 学習時: マッピングを作成
                unique_vals = df[col_name].unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                self.group_mappings[group_name] = mapping
            else:
                # 推論時: 既存のマッピングを使用
                mapping = self.group_mappings.get(group_name, {})
            
            num_group_nodes = len(mapping)
            
            # エッジ追加
            for idx, val in enumerate(df[col_name]):
                if val in mapping:
                    listing_idx = idx
                    virtual_idx = current_offset + mapping[val]
                    
                    # 双方向エッジ
                    src.extend([listing_idx, virtual_idx])
                    dst.extend([virtual_idx, listing_idx])
            
            # オフセット更新
            current_offset += num_group_nodes
            total_virtual_nodes += num_group_nodes
            
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # 仮想ノードの特徴量初期化 (ゼロ)
        virtual_features = torch.zeros((total_virtual_nodes, x.shape[1]))
        x_all = torch.cat([x, virtual_features], dim=0)
        
        return Data(x=x_all, edge_index=edge_index).to(self.device), N

    def train(self, df, target_col, continuous_cols=[], categorical_cols=[], group_cols={}, epochs=50):
        """
        Args:
            df: 学習データ
            target_col: 予測対象のカラム名 (Y)
            continuous_cols: GNNの入力とする連続値カラム名のリスト
            categorical_cols: GNNの入力とするカテゴリカルカラム名のリスト
            group_cols: グラフ構造を作るための辞書 {'役割名': 'カラム名'}
                        例: {'local': 'neighbourhood', 'segment': 'room_type'}
        """
        self.cont_cols = continuous_cols
        self.cat_cols = categorical_cols
        self.group_cols = group_cols
        
        df_clean = df.reset_index(drop=True)
        data, num_listings = self._build_graph(df_clean, is_training=True)
        y = torch.tensor(df_clean[target_col].values, dtype=torch.float).to(self.device)
        
        # モデル定義 (GraphSAGE)
        class VirtualNodeGNN(torch.nn.Module):
            def __init__(self, in_channels, hidden, out_emb):
                super().__init__()
                self.conv1 = SAGEConv(in_channels, hidden)
                self.conv2 = SAGEConv(hidden, hidden)
                self.emb_head = Linear(hidden, out_emb)
                self.pred_head = Linear(out_emb + in_channels, 1)

            def forward(self, x, edge_index, num_listings):
                h = self.conv1(x, edge_index).relu()
                h = self.conv2(h, edge_index).relu()
                
                # 実データノードのみ取り出し
                h_listings = h[:num_listings]
                embedding = self.emb_head(h_listings)
                
                # 特徴量と埋め込みを結合して予測
                self_feat = x[:num_listings]
                out = self.pred_head(torch.cat([self_feat, embedding], dim=1))
                return out, embedding

        self.model = VirtualNodeGNN(data.num_features, self.hidden_channels, self.out_channels).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out, _ = self.model(data.x, data.edge_index, num_listings)
            loss = F.mse_loss(out.squeeze(), y)
            loss.backward()
            optimizer.step()
        self.model.eval()

    def get_embeddings(self, df):
        df_clean = df.reset_index(drop=True)
        # 推論モード (is_training=False) でグラフ構築
        data, num_listings = self._build_graph(df_clean, is_training=False)
        
        with torch.no_grad():
            _, embeddings = self.model(data.x, data.edge_index, num_listings)
            
        cols = [f"GNN_AGG_{i}" for i in range(self.out_channels)]
        return pd.DataFrame(embeddings.cpu().numpy(), columns=cols, index=df_clean.index)