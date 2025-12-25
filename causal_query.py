import numpy as np, pandas as pd
import networkx as nx
from dowhy import gcm


class CausalQuery:

    def __init__(self):
        self.causal_model = None  # 集約属性を含まない因果モデル
        self.ex_causal_model = None  # 集約属性を含む因果モデル
        self.causal_graph = nx.DiGraph()
        self.ex_causal_graph = nx.DiGraph()
        self.ex_edges = []
        self.groupby_col = ''
        self.agg_func = ''

    def causal_schema(self):
        return True


    def set_causal_graph(self, edges:list, ex_edges:list, groupby_col:any, agg_func:str):
        # print(f"edges: {edges}, ex_edges: {ex_edges}")
        # 因果グラフを作成
        self.causal_graph = nx.DiGraph(edges)
        # 拡張因果グラフを作成
        agg_nodes = []
        for attr1, attr2 in ex_edges:
            agg_nodes.append((f"{agg_func}_{attr1}", attr2))
        agg_edges = edges+agg_nodes
        # print(agg_edges)
        self.ex_causal_graph = nx.DiGraph(agg_edges)
        self.ex_edges = ex_edges
        self.groupby_col = groupby_col
        self.agg_func = agg_func
    

    def train_causal_model(self, df, ex_df):
        # 集約属性を含まないモデルを学習
        self.causal_model = gcm.ProbabilisticCausalModel(self.causal_graph)
        gcm.auto.assign_causal_mechanisms(self.causal_model, df)
        gcm.fit(self.causal_model, df)

        # 集約属性を含むモデルを学習
        self.ex_causal_model = gcm.ProbabilisticCausalModel(self.ex_causal_graph)
        print(gcm.auto.assign_causal_mechanisms(self.ex_causal_model, ex_df))
        gcm.fit(self.ex_causal_model, ex_df)

    
    @staticmethod
    def get_agg_attr(df):
        agg_attrs = []
        for attr in df.columns():
            if "POST_" in attr:
                agg_attrs.append(attr)
        return agg_attrs


    @staticmethod
    def get_external_nodes(ex_edges:list):
        ex_nodes = []
        for start, end in ex_edges:
            ex_nodes.append(start)
        return ex_nodes
    

    @staticmethod
    def get_descendants(graph:nx.DiGraph, intervened_attrs:list):
        descendants = []
        for attr in intervened_attrs:
            descendants.append(attr)
            descendants.extend([node for node in nx.descendants(G=graph, source=attr)])
        return descendants


    def what_if(self, df, interventions:dict):
        new_df = df.copy()
        # print(f"input df:\n{new_df}")

        # 集約属性の元の属性が介入の影響を受ける場合
        ## 因果グラフにおける介入対象ノードの下流に集約元のノードがある
        ex_nodes = self.get_external_nodes(self.ex_edges)
        affected_nodes = self.get_descendants(self.causal_graph, interventions.keys())
        # print(f"外部タプルに影響を与える属性: {ex_nodes}")
        # print(f"介入属性から影響を受ける属性: {affected_nodes}")
        affected_agg_nodes = list(set(ex_nodes) & set(affected_nodes))
        post_attrs = [f"POST_{node}" for node in affected_agg_nodes]  #
        agg_attrs =  [f"{self.agg_func}_{node}" for node in affected_agg_nodes]
        # print(affected_agg_nodes)
        if bool(affected_agg_nodes):
            # 1段階目の介入を実行
            ## 集約属性がないバージョンの因果モデルを用いる
            new_df = new_df.drop(agg_attrs, axis=1) # 集約属性を削除
            first_samples = gcm.interventional_samples(self.causal_model, interventions=interventions, observed_data=new_df)
            print(f"df after fist intervention:\n{first_samples}")

            # 集約属性の値を再計算
            new_df[affected_agg_nodes] = first_samples[post_attrs]
            ex_first_samples = self.extend_dataset(new_df)  # POST_attrを元に集約属性agg_POST_attrを計算
            ex_first_samples[affected_agg_nodes] = first_samples[affected_agg_nodes]
            print(f"df re:extecded:\n{ex_first_samples}")

            # 2段階目の介入を実行
            samples = gcm.interventional_samples(self.ex_causal_model, interventions=interventions, observed_data=ex_first_samples)
            print(f"df after second intervention:\n{samples}")

        # 集約属性の元の属性が介入の影響を受けない場合
        else:
            samples = gcm.interventional_samples(self.ex_causal_model, interventions=interventions, observed_data=new_df)

        return samples


    # DataFrame 内の外部タプルに影響を与える属性(self.ex_edges) の集約属性を追加する
    def extend_dataset(self, df:pd.DataFrame, blockcol:str=''):
        """
        大規模データセット向けに最適化された集約属性追加関数。
        `min` および `max` にも対応。
        
        Parameters:
            df (pd.DataFrame): 元のデータフレーム
            ex_edges (list of tuple): 外部依存関係の定義 [(外部属性A, 内部属性B), ...]
            group_by_col (str): 集約のスコープを指定する列（例: 'Category'）
            agg_func (str): 集約関数（例: 'mean', 'sum', 'min', 'max'])
        
        Returns:
            pd.DataFrame: 集約属性を追加したデータフレーム
        """
        # 元のデータフレームをコピー
        new_df = df.copy()
        ex_nodes = self.get_external_nodes(self.ex_edges)
        for external_attr in ex_nodes:
            # 新しい列名を生成
            aggregated_col_name = f"{self.agg_func}_{external_attr}"

            if blockcol == '':
                if self.agg_func in ['mean', 'sum', 'count']:
                    # グループごとに合計値と要素数を計算
                    group_stats = (
                        new_df.groupby(self.groupby_col)[external_attr]
                        .agg(['sum', 'count'])
                        .rename(columns={'sum': 'group_sum', 'count': 'group_count'})
                    )
                    # 元のデータフレームに統合
                    new_df = new_df.merge(
                        group_stats,
                        how='left',
                        left_on=self.groupby_col,
                        right_index=True
                    )
                    
                    # 自身を除いた集約値を計算
                    if self.agg_func == 'mean':
                        new_df['group_agg'] = new_df.apply(
                            lambda row: (row['group_sum'] - row[external_attr]) / (row['group_count'] - 1)
                            if row['group_count'] > 1 else row['group_sum'] - row[external_attr],
                            axis=1
                        )
                    elif self.agg_func == 'sum':
                        new_df['group_agg'] = new_df.apply(
                            lambda row: row['group_sum'] - row[external_attr],
                            axis=1
                        )
                    elif self.agg_func == 'count':
                        new_df['group_agg'] = new_df['group_count']
                    
                    # 自身を除いた集約値を列に設定
                    new_df[aggregated_col_name] = new_df['group_agg']
                    new_df.drop(columns=['group_sum', 'group_count', 'group_agg'], inplace=True)
                
                elif self.agg_func in ['min', 'max']:
                    group_sorted = (
                        new_df.groupby(self.groupby_col)[external_attr]
                        .apply(lambda x: sorted(x))
                    )

                    # グループごとに最小値・2番目の最小値、最大値・2番目の最大値を抽出
                    group_stats = group_sorted.apply(
                        lambda x: {
                            'min': x[0],
                            'second_min': x[1] if len(x) > 1 else None,
                            'max': x[-1],
                            'second_max': x[-2] if len(x) > 1 else None
                        }
                    ).apply(pd.Series)

                    # 元のデータフレームに統合
                    new_df = new_df.merge(
                        group_stats,
                        how='left',
                        left_on=self.groupby_col,
                        right_index=True
                    )

                    # 自身を除いた値の計算
                    if self.agg_func == 'min':
                        new_df[f"{self.agg_func}_{external_attr}"] = new_df.apply(
                            lambda row: row['second_min'] if row[external_attr] == row['min'] else row['min'],
                            axis=1
                        )
                    elif self.agg_func == 'max':
                        new_df[f"{self.agg_func}_{external_attr}"] = new_df.apply(
                            lambda row: row['second_max'] if row[external_attr] == row['max'] else row['max'],
                            axis=1
                        )
                    
                    # 不要な列を削除
                    new_df.drop(columns=['min', 'second_min', 'max', 'second_max'], inplace=True)
                
            else:
                if self.agg_func in ['min', 'max']:
                    raise ValueError("agg_func ['min', 'max'] with blockcol is not implemented.")
                # 各グループ（groupby_col）内で、store ごとの集計を事前計算
                group_stats = (
                    new_df.groupby([self.groupby_col, blockcol])[external_attr]
                    .agg(['sum', 'count'])  # 必要な統計量を計算
                    .rename(columns={'sum': 'group_sum', 'count': 'group_count'})
                    .reset_index()
                )

                # グループ全体の統計量を計算
                total_stats = (
                    new_df.groupby(self.groupby_col)[external_attr]
                    .agg(['sum', 'count'])  # 全グループ内の統計量を計算
                    .rename(columns={'sum': 'total_sum', 'count': 'total_count'})
                    .reset_index()
                )

                # グループ全体の統計量をマージ
                group_stats = group_stats.merge(total_stats, on=self.groupby_col, how='left')

                # 自分以外のタプルの値を計算（同じstoreを除外）
                if self.agg_func == 'mean':
                    group_stats['group_agg'] = group_stats.apply(
                        lambda row: (row['total_sum'] - row['group_sum']) / (row['total_count'] - row['group_count'])
                        if row['total_count'] > row['group_count'] else None,
                        axis=1
                    )
                elif self.agg_func == 'sum':
                    group_stats['group_agg'] = group_stats.apply(
                        lambda row: row['total_sum'] - row['group_sum'],
                        axis=1
                    )

                # 計算済みの統計量を元のデータフレームにマージ
                new_df = new_df.merge(
                    group_stats[[self.groupby_col, blockcol, 'group_agg']],
                    on=[self.groupby_col, blockcol],
                    how='left'
                )

                new_df[aggregated_col_name] = new_df['group_agg']
                new_df.drop(columns=['group_agg'], inplace=True)
        
        return new_df

