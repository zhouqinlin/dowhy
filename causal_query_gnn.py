import pandas as pd
import networkx as nx
from dowhy import gcm
from gnn_aggregator import GenericGNN

class CausalQueryGNN:
    def __init__(self):
        self.causal_model = None
        self.gnn = None 
        self.agg_cols = []
        self.target_col = None
        
    def train(self, df, target_col, continuous_cols, categorical_cols, group_cols, causal_graph=None, gnn_hidden=64, gnn_out=16):
        """
        causal_graph (nx.DiGraph): ユーザー定義の因果グラフ。
                                   ノード名は df のカラム名と一致している必要があります。
        """
        print("Training Generic GNN Aggregator...")
        self.target_col = target_col
        self.gnn = GenericGNN(hidden_channels=gnn_hidden, out_channels=gnn_out)
        
        # 1. GNNの学習 (表現学習)
        # GNNは「近隣情報」を要約してターゲットを予測するように学習します
        self.gnn.train(
            df, 
            target_col=target_col,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            group_cols=group_cols,
            epochs=50
        )
        
        # 2. 埋め込みベクトルの取得
        agg_df = self.gnn.get_embeddings(df)
        self.agg_cols = agg_df.columns.tolist()
        df_extended = pd.concat([df.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)
        
        # 3. 因果グラフの構築 (厳密化)
        if causal_graph is None:
            # デフォルト: フラット構造 (非推奨だが互換性のため残す)
            feature_nodes = continuous_cols + categorical_cols
            nodes = feature_nodes + [target_col] + self.agg_cols
            edges = []
            for col in feature_nodes:
                edges.append((col, target_col))
            final_graph = nx.DiGraph(edges)
        else:
            # ユーザー定義グラフを使用
            final_graph = causal_graph.copy()
        
        # GNN埋め込みノードをグラフに追加
        # 仮定: GNN埋め込み(近隣の状況)は、ターゲット変数に直接影響を与える「未観測の交絡因子/コンテキスト」として扱う
        for col in self.agg_cols:
            final_graph.add_edge(col, target_col)
            
        # 4. 因果モデルの学習 (DoWhy GCM)
        self.causal_model = gcm.ProbabilisticCausalModel(final_graph)
        gcm.auto.assign_causal_mechanisms(self.causal_model, df_extended)
        gcm.fit(self.causal_model, df_extended)

    def predict(self, df):
        aggs = self.gnn.get_embeddings(df)
        gcm_input = pd.concat([df.reset_index(drop=True), aggs.reset_index(drop=True)], axis=1)
        
        mechanism = self.causal_model.causal_mechanism(self.target_col)
        # 親ノードの順序をグラフ定義と一致させる
        parents = sorted([p for p in self.causal_model.graph.predecessors(self.target_col)])
        parent_data = gcm_input[parents].to_numpy()
        
        samples = mechanism.draw_samples(parent_data)
        return samples.flatten()

    def what_if(self, df, interventions):
        """
        介入後（Post-intervention）の期待値を計算する。
        反事実推論ステップとして、観測データのノイズを保ったまま
        「平均的な変動量」だけを計算するために使用する。
        """
        intervened_df = df.copy().reset_index(drop=True)
        
        # 1. 介入の適用 (Rootノードへの介入)
        for node, action in interventions.items():
            condition = action.get('condition', lambda x: True)
            func = action.get('intervention')
            mask = intervened_df.apply(condition, axis=1)
            if mask.any():
                intervened_df.loc[mask, node] = intervened_df.loc[mask, node].apply(func)
        
        # 2. GNN埋め込みの更新 (波及効果の計算)
        # 価格が変われば、「近隣の価格相場(GNN埋め込み)」も変化する
        new_aggs = self.gnn.get_embeddings(intervened_df)
        gcm_input = pd.concat([intervened_df, new_aggs], axis=1)
        
        # 3. 期待値の伝播 (Counterfactual Mean Propagation)
        # グラフの構造に従って、介入の影響を受けたノードの「期待値」を更新していく
        # これにより、中間変数がある場合も整合性が取れる
        
        return self._predict_expectation(gcm_input, self.target_col, interventions)

    def _predict_expectation(self, data, target_node, interventions=None):
        """
        グラフのトポロジカル順序に従い、期待値(ノイズなし)を伝播させて予測するヘルパー関数
        """
        # データのコピーを作成
        current_data = data.copy()
        
        # ターゲットの祖先を特定
        ancestors = nx.ancestors(self.causal_model.graph, target_node)
        topo_order = list(nx.topological_sort(self.causal_model.graph))
        
        for node in topo_order:
            # ターゲットノード以降は計算不要（今回はターゲットまでで良い）
            if node == target_node:
                # ターゲットの親の値を使って期待値を予測
                mech = self.causal_model.causal_mechanism(node)
                parents = sorted([p for p in self.causal_model.graph.predecessors(node)])
                parent_data = current_data[parents].to_numpy()
                
                # NOTE: DoWhyのメカニズムによっては predict が未実装の場合があるため分岐
                try:
                    # 多くのGCMメカニズム(AdditiveNoiseModelなど)は predict で期待値を返す
                    pred = mech.predict(parent_data).flatten()
                except:
                    # predictがない場合は draw_samples で代用するが、ノイズが乗ってしまう
                    # 可能なら平均を取るなどの対策が必要だが、一旦そのまま
                    pred = mech.draw_samples(parent_data).flatten()
                
                return pred

            # 介入変数やGNN埋め込みは固定（すでに計算済み）
            if node in self.agg_cols: 
                continue
            if interventions and node in interventions:
                continue
            
            # 中間ノードの更新
            # もしこのノードが介入の影響を受けるなら、その期待値を計算してデータを更新する
            if node in ancestors:
                is_affected = False
                if interventions:
                    for intervention_node in interventions.keys():
                        if intervention_node in nx.ancestors(self.causal_model.graph, node):
                            is_affected = True
                            break
                
                if is_affected:
                    mech = self.causal_model.causal_mechanism(node)
                    parents = sorted([p for p in self.causal_model.graph.predecessors(node)])
                    if len(parents) > 0:
                        p_data = current_data[parents].to_numpy()
                        try:
                            # 期待値で更新（ノイズを含めない）
                            current_data[node] = mech.predict(p_data).flatten()
                        except:
                            current_data[node] = mech.draw_samples(p_data).flatten()
                            
        # 万が一ループを抜けた場合（通常ありえない）
        return current_data[target_node].to_numpy()