import numpy as np, pandas as pd
import networkx as nx
from dowhy import gcm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ttest_rel
import time

from causal_query import CausalQuery
from causal_query_gnn import CausalQueryGNN


def paired_ttest(df, attr, group_col="year"):
    results = []
    groups = df[group_col].unique()
    for group in groups:
        group_data = df[df[group_col] == group]
        if len(group_data) < 2:
            continue

        # 差分が全くない場合はスキップ（エラー回避）
        diff = group_data[f"POST_{attr}"] - group_data[attr]
        if diff.std() == 0:
            continue

        t_stat, p_value = ttest_rel(group_data[attr], group_data[f"POST_{attr}"])
        results.append({group_col: group, "t_stat": t_stat, "p_value": p_value})

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        significant_results = results_df[results_df["p_value"] < 0.05]
        print(f"\n有意な差が検出された {group_col}:")
        print(significant_results)
        return significant_results
    else:
        print(f"\n{group_col} ごとの検定結果はありません（データ不足の可能性）")
    return pd.DataFrame()


def evaluate_impact(df, result_col, significant_df, group_col="year"):
    post_col = f"POST_{result_col}"
    if post_col not in df.columns:
        return 0, 0, 0.0

    delta = df[post_col] - df[result_col]
    changed_mask = delta.abs() > 1e-6
    total_changed = changed_mask.sum()

    sig_groups = []
    if not significant_df.empty:
        sig_groups = significant_df[group_col].tolist()

    sig_mask = df[group_col].isin(sig_groups)
    detected_changed = (changed_mask & sig_mask).sum()
    ratio = detected_changed / total_changed if total_changed > 0 else 0.0
    return detected_changed, total_changed, ratio, delta


# Data Loading
try:
    df = pd.read_csv("datasets/abortion.csv")
except:
    print("Dataset not found. Please check the path.")
    exit()

df = df[["fip", "year", "repeal", "lnr", "poverty"]].dropna()
df["fip"] = df["fip"].astype(str)
df["year"] = df["year"].astype(int)

# Intervention: Legalize abortion (repeal=1)
treatment = "repeal"
outcome = "lnr"
interventions = {
    treatment: {
        "condition": lambda row: row[treatment] == 0,
        "intervention": lambda x: 1,
    }
}

print(f"Data Shape: {df.shape}")

# ==========================================
# 1. Conventional (DoWhy)
# ==========================================
print("\n--- Method 1: DoWhy ---")
causal_model = gcm.ProbabilisticCausalModel(
    nx.DiGraph(
        [
            ("fip", "repeal"), 
            ("year", "repeal"), 
            ("poverty", "lnr"), 
            ("fip", "lnr"), 
            ("year", "lnr"), 
            ("repeal", "lnr")
         ]
    )
)
gcm.auto.assign_causal_mechanisms(causal_model, df)
gcm.fit(causal_model, df)

mech = causal_model.causal_mechanism(outcome)
parents = sorted([p for p in causal_model.graph.predecessors(outcome)])

# 介入データの作成
df_in = df.copy()
mask = df_in.apply(interventions[treatment]["condition"], axis=1)
df_in.loc[mask, treatment] = 1
pred_post = mech.draw_samples(df_in[parents].to_numpy()).flatten()

# 結果の格納
df_conv = df.copy()
df_conv[f"POST_{outcome}"] = pred_post

sig_conv = paired_ttest(df_conv, outcome)
det_c, tot_c, rat_c, _ = evaluate_impact(df_conv, outcome, sig_conv)
print(f"[DoWhy] Updated: {tot_c}, Detected: {det_c}, Ratio: {rat_c:.2%}")

# ==========================================
# 2. Oiwa (Static)
# ==========================================
print("\n--- Method 2: Oiwa (Static) ---")
start_static = time.time()
cq_static = CausalQuery()
# Aggregate by Year (temporal spillover across states)
cq_static.set_causal_graph(
    [("fip", "repeal"), 
     ("year", "repeal"), 
     ("poverty", "lnr"), 
     ("fip", "lnr"), 
     ("year", "lnr"), 
     ("repeal", "lnr")],
    [(treatment, outcome)],
    "year",
    "mean",
)
ex_data = cq_static.extend_dataset(df, blockcol="fip")
cq_static.train_causal_model(df, ex_data)

model_s = cq_static.ex_causal_model
mech_s = model_s.causal_mechanism(outcome)
parents_s = sorted([p for p in model_s.graph.predecessors(outcome)])

df_in_s = df.copy()
df_in_s.loc[mask, treatment] = 1
ex_data_post = cq_static.extend_dataset(df_in_s, blockcol="fip")
pred_post_s = mech_s.draw_samples(ex_data_post[parents_s].to_numpy()).flatten()
end_static = time.time()

# 結果の格納
df_static = df.copy()
df_static[f"POST_{outcome}"] = pred_post_s

sig_static = paired_ttest(df_static, outcome)
det_s, tot_s, rat_s, _ = evaluate_impact(df_static, outcome, sig_static)
print(f"[Oiwa] Updated: {tot_s}, Detected: {det_s}, Ratio: {rat_s:.2%}")

# ==========================================
# 3. GNN (Proposed)
# ==========================================
print("\n--- Method 3: GNN (Generic) ---")

causal_graph = nx.DiGraph([
    ('fip', 'repeal'), 
    ('year', 'repeal'), 
    ('poverty', 'lnr'), 
    ('fip', 'lnr'), 
    ('year', 'lnr'), 
    ('repeal', 'lnr')
])
start_gnn_train = time.time()
cq_gnn = CausalQueryGNN()

# Config for Abortion
# Year is the bridge (temporal correlation across states)
cq_gnn.train(
    df,
    target_col="lnr",
    continuous_cols=["poverty", "repeal"],
    categorical_cols=["fip"],
    group_cols={"year_bridge": "year"},
    causal_graph=causal_graph,
    gnn_hidden=64,
)
end_gnn_train = time.time()

start_gnn_whatif = time.time()
pred_post_g = cq_gnn.what_if(df, interventions)
end_gnn_whatif = time.time()

# 結果の格納
df_gnn = df.copy()
df_gnn[f"POST_{outcome}"] = pred_post_g

sig_gnn = paired_ttest(df_gnn, outcome)
det_g, tot_g, rat_g, delta_g = evaluate_impact(df_gnn, outcome, sig_gnn)
print(f"[GNN] Updated: {tot_g}, Detected: {det_g}, Ratio: {rat_g:.2%}")
print(f"DEBUG: Delta Mean: {delta_g.mean():.6f}")

if not sig_gnn.empty:
    print(f"Significant Years:\n{sig_gnn.sort_values('p_value').head(5)}")

# ==========================================
# Visualization (Line Plot by Year)
# ==========================================
res_pre = df.groupby("year")[outcome].mean()
res_conv = df_conv.groupby("year")[f"POST_{outcome}"].mean()
res_static = df_static.groupby("year")[f"POST_{outcome}"].mean()
res_gnn = df_gnn.groupby("year")[f"POST_{outcome}"].mean()

plt.figure(figsize=(10, 5))
plt.plot(res_pre.index, res_pre, label="PRE (Observed)", marker="o", color="gray", linestyle="--")
plt.plot(res_conv.index, res_conv, label="DoWhy (Predicted)", marker="x")
plt.plot(res_static.index, res_static, label="AGG (Predicted)", marker="^")
plt.plot(res_gnn.index, res_gnn, label="GNN (Predicted)", marker="s", color="red")

plt.title("Abortion Legalization Impact")
plt.xlabel("Year")
plt.ylabel(outcome)
plt.legend()
plt.grid(True)
plt.savefig("exp_result/abortion_comparison.png")

print("\n************************\n実行時間の比較\n************************")
print(f"大岩手法 (Static): {end_static - start_static:.4f} sec")
print(f"改善手法 (GNN): Total {end_gnn_whatif - start_gnn_train:.4f} sec")
print(f"Graph saved to exp_result/abortion_comparison.png")
