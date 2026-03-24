import numpy as np, pandas as pd
import json
import networkx as nx
from dowhy import gcm
from causal_query import CausalQuery
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ttest_rel
import time
import torch

# 追加: 動的集約のためのモジュール
from dynamic_aggregator import DynamicAggregatorHandler

# 結果検証のためのt検定 attrとPOST_attrを比較
def paired_ttest(df, attr):
    results = []
    neighbourhoods = df["neighbourhood_cleansed"].unique()

    for neighbourhood in neighbourhoods:
        neighbourhood_data = df[df["neighbourhood_cleansed"] == neighbourhood]
        if len(neighbourhood_data) < 2:
            continue
        t_stat, p_value = ttest_rel(neighbourhood_data[attr], neighbourhood_data[f"POST_{attr}"])
        results.append({"neighbourhood_cleansed": neighbourhood, "t_stat": t_stat, "p_value": p_value})

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        significant_results = results_df[results_df["p_value"] < 0.05]
        print("\n有意な差が検出された neighbourhood_cleansed:")
        print(significant_results)
        return significant_results["neighbourhood_cleansed"].tolist()
    else:
        print("\n有意な差は検出されませんでした")
        return []
    
def evaluate_impact(df, result_col, significant_neighbourhoods):
    """
    更新されたタプルの中で、有意な差が検出されたタプルの割合を計算する
    """
    post_col = f"POST_{result_col}"
    if post_col not in df.columns:
        return 0, 0, 0.0

    # 1. 更新されたタプル (予測値が変化したもの)
    # 浮動小数点の誤差を考慮して 1e-5 以上の変化を対象とする
    changed_mask = (df[post_col] - df[result_col]).abs() > 1e-5
    total_changed = changed_mask.sum()

    # 2. 検出されたタプル (有意差ありと判定されたグループに属するもの)
    sig_mask = df["neighbourhood_cleansed"].isin(significant_neighbourhoods)
    detected_changed = (changed_mask & sig_mask).sum()

    # 3. 検出率 (Coverage/Recall的な指標)
    ratio = detected_changed / total_changed if total_changed > 0 else 0.0
    
    return detected_changed, total_changed, ratio

# --- CausalQueryを拡張して動的集約に対応させるクラス ---
class DynamicCausalQuery(CausalQuery):
    def __init__(self, aggregator_handler):
        super().__init__()
        self.handler = aggregator_handler

    # 親クラスの extend_dataset をオーバーライドして、Pyroモデルで集約値を計算する
    def extend_dataset(self, df, blockcol=''):
        # ハンドラーを使って動的集約列(DYNAMIC_price等)を追加
        # 注意: what_if内で再帰的に呼ばれるため、dfは介入後のデータ(POST_price等)を含む可能性がある
        
        # 介入後のデータ形式に対応するためのマッピング
        # CausalQueryのwhat_ifロジックでは、集約前の値が更新されている
        # ハンドラーが期待する列名に合わせる必要がある
        
        # 一時的にデータをコピー
        temp_df = df.copy()
        
        # ハンドラーを使って推論 (evalモード)
        # 内部でTensor化してAttention計算 -> 集約値生成
        # ここでは学習済みモデルを使用するだけなので高速
        extended_df = self.handler.append_dynamic_agg_feature(temp_df)
        
        target_agg_col = f"DYNAMIC_{self.handler.value_col}" # DYNAMIC_price_norm
        expected_dag_node = "DYNAMIC_price"                # DAGが期待する名前
        
        if target_agg_col in extended_df.columns and target_agg_col != expected_dag_node:
            extended_df = extended_df.rename(columns={target_agg_col: expected_dag_node})
        
        return extended_df

# ---------------------------------------------------------

# データセットの読み込み
df = pd.read_csv("datasets/airbnb_cleaned.csv")

# カラム選択 (緯度経度があればそれをコンテキストにするが、ここではカテゴリから擬似的にコンテキストを作る)
# もしデータセットに 'latitude', 'longitude' がある場合は target_cols に追加してください
target_cols = ["room_type", "review_scores_rating", "price", "neighbourhood_cleansed"]
# 仮に緯度経度がない場合、neighbourhoodを数値化してコンテキストにするための前処理
if "latitude" not in df.columns:
    df["neighbourhood_code"] = df["neighbourhood_cleansed"].astype('category').cat.codes
    df["room_type_code"] = df["room_type"].astype('category').cat.codes
    context_cols = ["neighbourhood_code", "room_type_code"]
    target_cols.extend(context_cols)
else:
    target_cols.extend(["latitude", "longitude"])
    context_cols = ["latitude", "longitude"]

df = df[target_cols].dropna()

# 価格の数値変換
if df["price"].dtype == "O":
    df["price"] = df["price"].replace("[\$,]", "", regex=True).astype(float)

df = df[df["review_scores_rating"] > 0]

print(f"Data Shape: {df.shape}")

# 分析対象の設定
target_room_type = "Entire home/apt"
target_neighbourhood = df[df["room_type"] == target_room_type]["neighbourhood_cleansed"].value_counts().idxmax()
print(f"Intervention Target Neighbourhood: {target_neighbourhood}")

# 介入条件: 特定地域の価格を30%下げる
interventions = {
    "price": {
        "condition": lambda row: row["neighbourhood_cleansed"] == target_neighbourhood
        and row["room_type"] == target_room_type,
        "intervention": lambda x: x * 0.70,
    }
}
result_val = "review_scores_rating"


# ==========================================
# 1. 従来のDoWhy (ベースライン)
# ==========================================
print("\n--- Method 1: DoWhy (Conventional) ---")
causal_graph = nx.DiGraph(
    [
        ("neighbourhood_cleansed", "room_type"),
        ("neighbourhood_cleansed", "price"),
        ("room_type", "price"),
        ("price", "review_scores_rating"),
        ("neighbourhood_cleansed", "review_scores_rating"),
    ]
)
causal_model = gcm.ProbabilisticCausalModel(causal_graph)
gcm.auto.assign_causal_mechanisms(causal_model, df)
gcm.fit(causal_model, df)

convresult = gcm.interventional_samples(causal_model, interventions, observed_data=df)
convresult = convresult.loc[convresult["room_type"] == target_room_type]

# 集計と検定
group_counts = convresult.groupby("neighbourhood_cleansed").size()
valid_neighbourhood = group_counts[group_counts >= 100].index
filtered_conv = convresult[convresult["neighbourhood_cleansed"].isin(valid_neighbourhood)]
sig_neighbourhoods_conv = paired_ttest(filtered_conv, result_val)
det_conv, tot_conv, rat_conv = evaluate_impact(filtered_conv, result_val, sig_neighbourhoods_conv)
print(f"[Conventional Result Analysis]")
print(f"  - Total Updated Tuples: {tot_conv}")
print(f"  - Detected in Sig Groups: {det_conv}")
print(f"  - Detection Coverage: {rat_conv:.2%}")

groupby_convresult = filtered_conv.groupby(["neighbourhood_cleansed"])[
    ["price", "POST_price", "review_scores_rating", "POST_review_scores_rating"]
].mean()


# ==========================================
# 2. 大岩手法 (静的集約: Mean of Group)
# ==========================================
print("\n--- Method 2: Ooiwa Method (Static Aggregation) ---")
start_static = time.time()
causal_query_static = CausalQuery()
agg_func = "mean"
groupby_col = "room_type"

# DAG: 他地域の価格平均(mean_price)が評価に影響
causal_query_static.set_causal_graph(
    [
        ("neighbourhood_cleansed", "room_type"),
        ("neighbourhood_cleansed", "price"),
        ("price", "review_scores_rating"),
        ("neighbourhood_cleansed", "review_scores_rating"),
    ],
    [("price", "review_scores_rating")], 
    groupby_col,
    agg_func,
)

# 静的拡張
ex_training_data_static = causal_query_static.extend_dataset(df, blockcol="neighbourhood_cleansed")
causal_query_static.train_causal_model(df, ex_training_data_static)
proresult = causal_query_static.what_if(ex_training_data_static, interventions)
end_static = time.time()

proresult = proresult[proresult["room_type"] == target_room_type]
filtered_pro = proresult[proresult["neighbourhood_cleansed"].isin(valid_neighbourhood)]
sig_neighbourhoods_pro = paired_ttest(filtered_pro, result_val)
det_pro, tot_pro, rat_pro = evaluate_impact(filtered_pro, result_val, sig_neighbourhoods_pro)
print(f"[Static Result Analysis]")
print(f"  - Total Updated Tuples: {tot_pro}")
print(f"  - Detected in Sig Groups: {det_pro}")
print(f"  - Detection Coverage: {rat_pro:.2%}")

groupby_proresult = filtered_pro.groupby(["neighbourhood_cleansed"])[
    ["price", "POST_price", "review_scores_rating", "POST_review_scores_rating"]
].mean()


# ==========================================
# 3. 改善手法 (動的集約: Pyro / Attention)
# ==========================================
print("\n--- Method 3: Improved Method (Dynamic Aggregation with Pyro) ---")
start_dynamic_train = time.time()

# 1. データ準備 (Embedding用にカテゴリと数値を分ける)
cat_cols = ["neighbourhood_cleansed", "room_type"]
cont_cols = []
if "latitude" in df.columns:
    # 緯度経度があれば正規化して連続変数として使う
    df["lat_norm"] = (df["latitude"] - df["latitude"].mean()) / df["latitude"].std()
    df["lon_norm"] = (df["longitude"] - df["longitude"].mean()) / df["longitude"].std()
    cont_cols = ["lat_norm", "lon_norm"]

# ターゲットの値も正規化
df["price_norm"] = (df["price"] - df["price"].mean()) / df["price"].std()

# Handler用のカテゴリコード作成などの準備はHandler内部で行われるが
# DataFrame上の型変換だけ確実にしておく
for c in cat_cols:
    if df[c].dtype == 'object':
        df[c] = df[c].astype('category')
        
df["room_type_code"] = df["room_type"].cat.codes

# Handlerの初期化 (引数が変わっています)
handler = DynamicAggregatorHandler(
    df=df,
    cat_context_cols=cat_cols,
    cont_context_cols=cont_cols,
    value_col="price_norm",
    own_feature_cols=["room_type_code"], # 事前に作成したコード列
    target_col="review_scores_rating",
    embedding_dim=16
)

# 学習 (Epoch数を増やして十分学習させる)
print("Training Pyro Attention Model (Embeddings)...")
handler.train(num_iterations=2000, lr=0.005, batch_size=1024)
end_dynamic_train = time.time()

# --- DynamicCausalQueryのセットアップ ---
start_dynamic_whatif = time.time()
dynamic_query = DynamicCausalQuery(handler)
dynamic_query.set_causal_graph(
    [
        ("neighbourhood_cleansed", "room_type"), ("neighbourhood_cleansed", "price"),
        ("price", "review_scores_rating"), ("neighbourhood_cleansed", "review_scores_rating"),
    ],
    [("price", "review_scores_rating")], 
    groupby_col=None,
    agg_func="DYNAMIC", 
)

ex_training_data_dynamic = dynamic_query.extend_dataset(df)
dynamic_query.train_causal_model(df, ex_training_data_dynamic)
dynamic_result = dynamic_query.what_if(ex_training_data_dynamic, interventions)
end_dynamic_whatif = time.time()

# 結果のフィルタリング
dynamic_result = dynamic_result[dynamic_result["room_type"] == target_room_type]
filtered_dynamic = dynamic_result[dynamic_result["neighbourhood_cleansed"].isin(valid_neighbourhood)]

# ★★★ 新しい評価指標の計算と表示 ★★★
sig_neighbourhoods_dynamic = paired_ttest(filtered_dynamic, result_val)
groupby_dynamic_result = filtered_dynamic.groupby(["neighbourhood_cleansed"])[
    ["price", "POST_price", "review_scores_rating", "POST_review_scores_rating"]
].mean()

det_dyn, tot_dyn, rat_dyn = evaluate_impact(filtered_dynamic, result_val, sig_neighbourhoods_dynamic)

print(f"[Dynamic Result Analysis]")
print(f"  - Total Updated Tuples: {tot_dyn}")
print(f"  - Detected in Sig Groups: {det_dyn}")
print(f"  - Detection Coverage: {rat_dyn:.2%}")
print(f"  - Significant Neighbourhoods: {sig_neighbourhoods_dynamic}")

# ==========================================
# 結果の可視化と出力
# ==========================================
qualified_neighbourhoods = list(
    set(sig_neighbourhoods_conv) | set(sig_neighbourhoods_pro) | set(sig_neighbourhoods_dynamic)
)

# プロット用データの整形
top_n = 15
valid_counts = df[df["neighbourhood_cleansed"].isin(qualified_neighbourhoods)]["neighbourhood_cleansed"].value_counts()
final_target_neighbourhoods = valid_counts.head(top_n).index

# 各結果をフィルタリング
plot_conv = groupby_convresult[groupby_convresult.index.isin(final_target_neighbourhoods)].reindex(final_target_neighbourhoods)
plot_static = groupby_proresult[groupby_proresult.index.isin(final_target_neighbourhoods)].reindex(final_target_neighbourhoods)
plot_dynamic = groupby_dynamic_result[groupby_dynamic_result.index.isin(final_target_neighbourhoods)].reindex(final_target_neighbourhoods)

neighbourhoods = final_target_neighbourhoods
x = np.arange(len(neighbourhoods))
bar_width = 0.2

# グラフ描画
plt.figure(figsize=(12, 6))
rcParams.update({"font.size": 10})

# 元の値(PRE)はStaticの結果から参照(どれでも同じ)
plt.bar(x - bar_width, plot_static[result_val], width=bar_width, label=f"PRE {result_val}", color="gray", alpha=0.5)
plt.bar(x, plot_conv[f"POST_{result_val}"], width=bar_width, label="POST (DoWhy)", color="tab:blue")
plt.bar(x + bar_width, plot_static[f"POST_{result_val}"], width=bar_width, label="POST (Static Agg)", color="tab:orange")
plt.bar(x + 2*bar_width, plot_dynamic[f"POST_{result_val}"], width=bar_width, label="POST (Dynamic Agg)", color="tab:red")

plt.title(f"Comparison of Intervention Effects: {result_val}")
plt.xlabel("Neighbourhood")
plt.xticks(x + bar_width/2, neighbourhoods, rotation=45, ha="right")
plt.ylabel(result_val)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(f"exp_result/comparison_airbnb_dynamic.png", dpi=300)
plt.close()

# 結果表示
print("\n************************\n実行時間の比較\n************************")
print(f"大岩手法 (Static): {end_static - start_static:.4f} sec")
print(f"改善手法 (Dynamic): Total {end_dynamic_whatif - start_dynamic_train:.4f} sec")
print(f"  - Training: {end_dynamic_train - start_dynamic_train:.4f} sec")
print(f"  - What-If: {end_dynamic_whatif - start_dynamic_whatif:.4f} sec")

print("\n************************\n数値結果 (Top 5 Areas)\n************************")
print(f"更新前 (PRE):\n{plot_static[result_val].head().values}")
print(f"更新後 (DoWhy):\n{plot_conv[f'POST_{result_val}'].head().values}")
print(f"更新後 (Static):\n{plot_static[f'POST_{result_val}'].head().values}")
print(f"更新後 (Dynamic):\n{plot_dynamic[f'POST_{result_val}'].head().values}")

print("\nExperiment Completed. Graph saved to exp_result/comparison_airbnb_dynamic.png")