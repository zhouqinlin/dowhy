import numpy as np, pandas as pd



def verify_extended_dataset(df: pd.DataFrame, extended_df: pd.DataFrame, groupby_col: str, external_attr: str, agg_func: str, blockcol:str):
    """
    拡張データセットが正しく計算されているかを検証します。
    
    Parameters:
        df (pd.DataFrame): 元のデータフレーム
        extended_df (pd.DataFrame): 拡張後のデータフレーム
        groupby_col (str): グループ化に使用する列
        external_attr (str): 集約対象の列
        agg_func (str): 集約関数（例: 'mean', 'sum', 'min', 'max'）
    """
    print(f"検証を開始します（グループ化列: {groupby_col}, 属性: {external_attr}, 集約関数: {agg_func}）")
    
    # 拡張データの列名
    aggregated_col_name = f"{agg_func}_{external_attr}"

    # 各行について検証
    for index, row in df.iterrows():
        # 元データからグループ内で同じ groupby_col のタプルを抽出
        group_data = df[df[groupby_col] == row[groupby_col]]
        
        # 自分と異なる store を持つタプルをフィルタリング
        filtered_data = group_data[group_data[blockcol] != row[blockcol]]
        
        # 集約値を計算
        if agg_func == 'mean':
            expected_value = filtered_data[external_attr].mean() if not filtered_data.empty else None
        elif agg_func == 'sum':
            expected_value = filtered_data[external_attr].sum() if not filtered_data.empty else None
        elif agg_func == 'min':
            expected_value = filtered_data[external_attr].min() if not filtered_data.empty else None
        elif agg_func == 'max':
            expected_value = filtered_data[external_attr].max() if not filtered_data.empty else None
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")
        
        # 拡張データセットの値を取得
        actual_value = extended_df.at[index, aggregated_col_name]
        
        # 比較
        if pd.isna(expected_value) and pd.isna(actual_value):
            pass  # 両方とも欠損値ならOK
        elif not np.isclose(expected_value, actual_value, equal_nan=True):
            print(f"エラー: インデックス {index} の計算が一致しません")
            print(f"タプル: {df.loc[index]}")
            print(f"  期待される値: {expected_value}")
            print(f"  実際の値: {actual_value}")
            return
    
    print("すべての行が正しく計算されています！")