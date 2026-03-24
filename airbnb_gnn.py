import numpy as np, pandas as pd
import networkx as nx
from dowhy import gcm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ttest_rel
import time

from causal_query import CausalQuery
from causal_query_gnn import CausalQueryGNN

def paired_ttest(df, attr):
    results = []
    neighbourhoods = df["neighbourhood_cleansed"].unique()

    for neighbourhood in neighbourhoods:
        neighbourhood_data = df[df["neighbourhood_cleansed"] == neighbourhood]
        # Skip if no variance or too small
        if len(neighbourhood_data) < 2: continue
        
        # Check if values are identical (t-test fails if variance of diff is 0)
        diff = neighbourhood_data[f"POST_{attr}"] - neighbourhood_data[attr]
        if diff.std() == 0:
            continue
            
        t_stat, p_value = ttest_rel(neighbourhood_data[attr], neighbourhood_data[f"POST_{attr}"])
        results.append({"neighbourhood_cleansed": neighbourhood, "t_stat": t_stat, "p_value": p_value})

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        significant_results = results_df[results_df["p_value"] < 0.05]
        print("\n有意な差が検出された neighbourhood_cleansed:")
        print(significant_results)
        return significant_results
    else:
        print("\n有意な差は検出されませんでした（データ不足の可能性あり）")
        return pd.DataFrame()

def evaluate_impact(df, result_col, significant_df):
    post_col = f"POST_{result_col}"
    if post_col not in df.columns: return 0, 0, 0.0
    
    # Check Delta Magnitude
    delta = df[post_col] - df[result_col]
    
    changed_mask = delta.abs() > 1e-5
    total_changed = changed_mask.sum()

    sig_neighbourhoods = []
    if not significant_df.empty:
        sig_neighbourhoods = significant_df["neighbourhood_cleansed"].tolist()
        
    sig_mask = df["neighbourhood_cleansed"].isin(sig_neighbourhoods)
    detected_changed = (changed_mask & sig_mask).sum()
    ratio = detected_changed / total_changed if total_changed > 0 else 0.0
    return detected_changed, total_changed, ratio, delta

def apply_imputation(df, model_pre, model_post, target_col):
    delta = model_post - model_pre
    return df[target_col] + delta

try:
    df = pd.read_csv("datasets/airbnb_cleaned.csv")
except:
    print("データセットが見つかりません。")
    exit()

target_cols = ["room_type", "review_scores_rating", "price", "neighbourhood_cleansed"]
df = df[target_cols].dropna()
if df["price"].dtype == "O":
    df["price"] = df["price"].replace("[\$,]", "", regex=True).astype(float)
df = df[df["review_scores_rating"] > 0]

target_room_type = "Entire home/apt"
target_neighbourhood = df[df["room_type"] == target_room_type]["neighbourhood_cleansed"].value_counts().idxmax()
print(f"Intervention Target Neighbourhood: {target_neighbourhood}")

interventions = {
    "price": {
        "condition": lambda row: row["neighbourhood_cleansed"] == target_neighbourhood
        and row["room_type"] == target_room_type,
        "intervention": lambda x: x * 0.70,
    }
}
result_val = "review_scores_rating"

# ==========================================
# 1. Conventional Method (DoWhy)
# ==========================================
print("\n--- Method 1: DoWhy (Conventional) ---")
causal_graph = nx.DiGraph([
    ("neighbourhood_cleansed", "room_type"),
    ("neighbourhood_cleansed", "price"),
    ("room_type", "price"),
    ("price", "review_scores_rating"),
    ("neighbourhood_cleansed", "review_scores_rating"),
])
model_conv = gcm.ProbabilisticCausalModel(causal_graph)
gcm.auto.assign_causal_mechanisms(model_conv, df)
gcm.fit(model_conv, df)

mech_rating = model_conv.causal_mechanism(result_val)
parents = sorted([p for p in model_conv.graph.predecessors(result_val)])
parent_data_pre = df[parents].to_numpy()
pred_pre_conv = mech_rating.draw_samples(parent_data_pre).flatten()

df_post_in = df.copy()
mask = df_post_in.apply(interventions['price']['condition'], axis=1)
df_post_in.loc[mask, 'price'] = df_post_in.loc[mask, 'price'] * 0.70
parent_data_post = df_post_in[parents].to_numpy()
pred_post_conv = mech_rating.draw_samples(parent_data_post).flatten()

df_conv_res = df.copy()
df_conv_res[f"POST_{result_val}"] = apply_imputation(df, pred_pre_conv, pred_post_conv, result_val)
df_conv_res = df_conv_res[df_conv_res["room_type"] == target_room_type]

group_counts = df_conv_res.groupby("neighbourhood_cleansed").size()
valid_neighs = group_counts[group_counts >= 100].index
df_conv_filtered = df_conv_res[df_conv_res["neighbourhood_cleansed"].isin(valid_neighs)]

sig_conv_df = paired_ttest(df_conv_filtered, result_val)
det_conv, tot_conv, rat_conv, _ = evaluate_impact(df_conv_filtered, result_val, sig_conv_df)
print(f"[Conventional Analysis]")
print(f"  - Updated Tuples: {tot_conv}")
print(f"  - Detected in Sig: {det_conv}")
print(f"  - Coverage: {rat_conv:.2%}")

# ==========================================
# 2. Oiwa's Method (Static)
# ==========================================
print("\n--- Method 2: Ooiwa Method (Static) ---")
start_static = time.time()
cq_static = CausalQuery()
agg_func = "mean"
groupby_col = "room_type"

cq_static.set_causal_graph(
    [("neighbourhood_cleansed", "room_type"), 
     ("neighbourhood_cleansed", "price"),
     ("price", "review_scores_rating"), 
     ("neighbourhood_cleansed", "review_scores_rating")],
    [("price", "review_scores_rating")], 
    groupby_col, agg_func,
)

ex_data = cq_static.extend_dataset(df, blockcol="neighbourhood_cleansed")
cq_static.train_causal_model(df, ex_data)

model_static = cq_static.ex_causal_model
mech_static = model_static.causal_mechanism(result_val)
parents_static = sorted([p for p in model_static.graph.predecessors(result_val)])
parent_data_static_pre = ex_data[parents_static].to_numpy()
pred_pre_static = mech_static.draw_samples(parent_data_static_pre).flatten()

df_post_static = df.copy()
mask = df_post_static.apply(interventions['price']['condition'], axis=1)
df_post_static.loc[mask, 'price'] = df_post_static.loc[mask, 'price'] * 0.70
ex_data_post = cq_static.extend_dataset(df_post_static, blockcol="neighbourhood_cleansed")
parent_data_static_post = ex_data_post[parents_static].to_numpy()
pred_post_static = mech_static.draw_samples(parent_data_static_post).flatten()

end_static = time.time()

df_static_res = df.copy()
df_static_res[f"POST_{result_val}"] = apply_imputation(df, pred_pre_static, pred_post_static, result_val)
df_static_res = df_static_res[df_static_res["room_type"] == target_room_type]

df_static_filtered = df_static_res[df_static_res["neighbourhood_cleansed"].isin(valid_neighs)]
sig_static_df = paired_ttest(df_static_filtered, result_val)
det_static, tot_static, rat_static, _ = evaluate_impact(df_static_filtered, result_val, sig_static_df)
print(f"[Static Analysis]")
print(f"  - Updated Tuples: {tot_static}")
print(f"  - Detected in Sig: {det_static}")
print(f"  - Coverage: {rat_static:.2%}")

# ==========================================
# 3. Improved Method (GNN)
# ==========================================
print("\n--- Method 3: GNN Method (Proposed) ---")

causal_graph = nx.DiGraph([
    ("neighbourhood_cleansed", "room_type"),
    ("neighbourhood_cleansed", "price"),
    ("price", "review_scores_rating"),
    ("neighbourhood_cleansed", "review_scores_rating"),
])
start_gnn_train = time.time()
cq_gnn = CausalQueryGNN()
# カラム設定を辞書で定義
cq_gnn.train(
    df,
    target_col='review_scores_rating',          # 予測したい値 (Y)
    continuous_cols=['price'],                  # 数値特徴量 (X_cont)
    categorical_cols=['room_type'],             # カテゴリ特徴量 (X_cat)
    group_cols={                                # グラフのエッジ定義
        'local': 'neighbourhood_cleansed',      # 地域でつなぐ
        'segment': 'room_type'                  # 部屋タイプでつなぐ
    },
    gnn_hidden=64
)
end_gnn_train = time.time()

pred_pre_gnn = cq_gnn.predict(df)

start_gnn_whatif = time.time()
pred_post_gnn = cq_gnn.what_if(df, interventions)
end_gnn_whatif = time.time()

df_gnn_res = df.copy()
df_gnn_res[f"POST_{result_val}"] = apply_imputation(df, pred_pre_gnn, pred_post_gnn, result_val)

df_gnn_res = df_gnn_res[df_gnn_res["room_type"] == target_room_type]
df_gnn_filtered = df_gnn_res[df_gnn_res["neighbourhood_cleansed"].isin(valid_neighs)]

sig_gnn_df = paired_ttest(df_gnn_filtered, result_val)
det_gnn, tot_gnn, rat_gnn, delta_gnn = evaluate_impact(df_gnn_filtered, result_val, sig_gnn_df)

print(f"[GNN Analysis]")
print(f"  - Updated Tuples: {tot_gnn}")
print(f"  - Detected in Sig: {det_gnn}")
print(f"  - Coverage: {rat_gnn:.2%}")
print(f"  - DEBUG: Delta Mean: {delta_gnn.mean():.6f}, Std: {delta_gnn.std():.6f}, Max: {delta_gnn.max():.6f}")

if not sig_gnn_df.empty:
    print(f"  - Significant Neighbourhoods (Top 5):\n{sig_gnn_df.sort_values('p_value').head(5)}")

# ==========================================
# Visualization
# ==========================================
all_sig = set()
if not sig_conv_df.empty: all_sig.update(sig_conv_df["neighbourhood_cleansed"])
if not sig_static_df.empty: all_sig.update(sig_static_df["neighbourhood_cleansed"])
if not sig_gnn_df.empty: all_sig.update(sig_gnn_df["neighbourhood_cleansed"])

top_n = 15
plot_neighs = [target_neighbourhood] + [n for n in list(all_sig) if n != target_neighbourhood][:top_n]

def get_plot_data(df_filtered, neighs):
    grp = df_filtered[df_filtered["neighbourhood_cleansed"].isin(neighs)].groupby("neighbourhood_cleansed")
    return grp[f"POST_{result_val}"].mean().reindex(neighs)

pre_mean = df[df["neighbourhood_cleansed"].isin(plot_neighs)].groupby("neighbourhood_cleansed")[result_val].mean().reindex(plot_neighs)
post_conv = get_plot_data(df_conv_filtered, plot_neighs)
post_static = get_plot_data(df_static_filtered, plot_neighs)
post_gnn = get_plot_data(df_gnn_filtered, plot_neighs)

x = np.arange(len(plot_neighs))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - 1.5*width, pre_mean, width, label="PRE (Observed)", color="gray", alpha=0.5)
plt.bar(x - 0.5*width, post_conv, width, label="POST (DoWhy)", color="tab:blue")
plt.bar(x + 0.5*width, post_static, width, label="POST (Oiwa)", color="tab:orange")
plt.bar(x + 1.5*width, post_gnn, width, label="POST (GNN)", color="tab:red")

plt.title(f"Comparison of Intervention Effects: {result_val}")
plt.xlabel("Neighbourhood")
plt.xticks(x, plot_neighs, rotation=45, ha="right")
plt.ylabel(result_val)
plt.legend()
plt.tight_layout()
plt.savefig("exp_result/airbnb_comparison.png")

print("\n************************\n実行時間の比較\n************************")
print(f"大岩手法 (Static): {end_static - start_static:.4f} sec")
print(f"改善手法 (GNN): Total {end_gnn_whatif - start_gnn_train:.4f} sec")