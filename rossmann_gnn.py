import numpy as np, pandas as pd
import networkx as nx
from dowhy import gcm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ttest_rel
import time
import os

from causal_query import CausalQuery
from causal_query_gnn import CausalQueryGNN


# --- Helper Functions ---
def paired_ttest(df, attr, group_col="Store"):
    results = []
    groups = df[group_col].unique()
    for group in groups:
        group_data = df[df[group_col] == group]
        if len(group_data) < 2:
            continue

        diff = group_data[f"POST_{attr}"] - group_data[attr]
        if diff.std() == 0:
            continue  # Skip if no variance

        t_stat, p_value = ttest_rel(group_data[attr], group_data[f"POST_{attr}"])
        results.append({group_col: group, "t_stat": t_stat, "p_value": p_value})

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        significant_results = results_df[results_df['p_value'] < 0.05]
        print("\n有意な差が検出された Store:")
        print(significant_results)
        return significant_results
    return pd.DataFrame()


def evaluate_impact(df, result_col, significant_df, group_col="Store"):
    post_col = f"POST_{result_col}"
    if post_col not in df.columns:
        return 0, 0, 0.0

    # Delta check
    delta = df[post_col] - df[result_col]
    changed_mask = delta.abs() > 1e-5
    total_changed = changed_mask.sum()

    sig_groups = []
    if not significant_df.empty:
        sig_groups = significant_df[group_col].tolist()

    sig_mask = df[group_col].isin(sig_groups)
    detected_changed = (changed_mask & sig_mask).sum()
    ratio = detected_changed / total_changed if total_changed > 0 else 0.0
    return detected_changed, total_changed, ratio, delta


def apply_imputation(df, model_pre, model_post, target_col):
    return df[target_col] + (model_post - model_pre)


# --- Data Loading ---
try:
    df = pd.read_csv("datasets/rossmann_store_sales.csv", low_memory=False)
except:
    print("Dataset not found.")
    exit()

# Preprocessing
df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
# Filter for speed (e.g., 2014 only)
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"].dt.year == 2014].copy()
df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")  # For grouping

target_cols = ["Store", "DateStr", "DayOfWeek", "Sales", "Customers", "Promo"]
df = df[target_cols].dropna()

df["Store"] = df["Store"].astype(str)
df["Promo"] = df["Promo"].astype(int)
df["Sales"] = df["Sales"].astype(float)

# Target: Friday
target_day = 5
result_val = "Sales"

interventions = {"Promo": {"condition": lambda row: row["DayOfWeek"] == target_day, "intervention": lambda x: 0}}

print(f"Data Shape: {df.shape}")

# ==========================================
# 1. Conventional (DoWhy)
# ==========================================
print("\n--- Method 1: DoWhy (Conventional) ---")
causal_graph = nx.DiGraph([("DayOfWeek", "Sales"), ("Promo", "Sales"), ("Store", "Sales")])
model_conv = gcm.ProbabilisticCausalModel(causal_graph)
gcm.auto.assign_causal_mechanisms(model_conv, df)
gcm.fit(model_conv, df)

mech = model_conv.causal_mechanism(result_val)
parents = sorted([p for p in model_conv.graph.predecessors(result_val)])
parent_data_pre = df[parents].to_numpy()
pred_pre = mech.draw_samples(parent_data_pre).flatten()

df_in = df.copy()
mask = df_in.apply(interventions["Promo"]["condition"], axis=1)
df_in.loc[mask, "Promo"] = 1
parent_data_post = df_in[parents].to_numpy()
pred_post = mech.draw_samples(parent_data_post).flatten()

df_conv = df.copy()
df_conv[f"POST_{result_val}"] = apply_imputation(df, pred_pre, pred_post, result_val)
df_conv = df_conv[df_conv["DayOfWeek"] == target_day]

sig_conv = paired_ttest(df_conv, result_val)
det_c, tot_c, rat_c, _ = evaluate_impact(df_conv, result_val, sig_conv)
print(f"[DoWhy] Updated: {tot_c}, Detected: {det_c}, Ratio: {rat_c:.2%}")

# ==========================================
# 2. Oiwa (Static)
# ==========================================
print("\n--- Method 2: Oiwa (Static) ---")
start_static = time.time()
cq_static = CausalQuery()
# Edge: DateStr connects stores on same day
cq_static.set_causal_graph(
    [("DayOfWeek", "Sales"), ("Promo", "Sales"), ("Store", "Sales")], [("Promo", "Sales")], "DateStr", "mean"
)
ex_data = cq_static.extend_dataset(df, blockcol="Store")
cq_static.train_causal_model(df, ex_data)

model_s = cq_static.ex_causal_model
mech_s = model_s.causal_mechanism(result_val)
parents_s = sorted([p for p in model_s.graph.predecessors(result_val)])
pred_pre_s = mech_s.draw_samples(ex_data[parents_s].to_numpy()).flatten()

df_in_s = df.copy()
df_in_s.loc[mask, "Promo"] = 1
ex_data_post = cq_static.extend_dataset(df_in_s, blockcol="Store")
pred_post_s = mech_s.draw_samples(ex_data_post[parents_s].to_numpy()).flatten()
end_static = time.time()

df_static = df.copy()
df_static[f"POST_{result_val}"] = apply_imputation(df, pred_pre_s, pred_post_s, result_val)
df_static = df_static[df_static["DayOfWeek"] == target_day]

sig_static = paired_ttest(df_static, result_val)
det_s, tot_s, rat_s, _ = evaluate_impact(df_static, result_val, sig_static)
print(f"[Oiwa] Updated: {tot_s}, Detected: {det_s}, Ratio: {rat_s:.2%}")

# ==========================================
# 3. GNN (Proposed)
# ==========================================
print("\n--- Method 3: GNN (Generic) ---")

causal_graph = nx.DiGraph([
    ('DayOfWeek', 'Sales'),
    ('Promo', 'Sales'),
    ('Store', 'Sales') # 店舗ごとのベースライン
])

start_gnn_train = time.time()
cq_gnn = CausalQueryGNN()

# Config for Rossmann
# DateStr is the bridge (temporal correlation)
cq_gnn.train(
    df,
    target_col="Sales",
    continuous_cols=["Promo"],
    categorical_cols=["DayOfWeek"],
    group_cols={"date_bridge": "DateStr"},
    causal_graph=causal_graph,
    gnn_hidden=64,
)
end_gnn_train = time.time()

pred_pre_g = cq_gnn.predict(df)
start_gnn_whatif = time.time()
pred_post_g = cq_gnn.what_if(df, interventions)
end_gnn_whatif = time.time()

df_gnn = df.copy()
df_gnn[f"POST_{result_val}"] = apply_imputation(df, pred_pre_g, pred_post_g, result_val)
df_gnn = df_gnn[df_gnn["DayOfWeek"] == target_day]

sig_gnn = paired_ttest(df_gnn, result_val)
det_g, tot_g, rat_g, delta_g = evaluate_impact(df_gnn, result_val, sig_gnn)
print(f"[GNN] Updated: {tot_g}, Detected: {det_g}, Ratio: {rat_g:.2%}")
print(f"DEBUG: Delta Mean: {delta_g.mean():.4f}")

if not sig_gnn.empty:
    print(f"Significant Stores (Top 5):\n{sig_gnn.sort_values('p_value').head(5)}")

# ==========================================
# Visualization
# ==========================================
all_sig = set()
if not sig_conv.empty:
    all_sig.update(sig_conv["Store"])
if not sig_static.empty:
    all_sig.update(sig_static["Store"])
if not sig_gnn.empty:
    all_sig.update(sig_gnn["Store"])

plot_stores = list(all_sig)[:15]
if not plot_stores:
    plot_stores = df["Store"].unique()[:15]


def get_plot_data(df_f, stores):
    return df_f[df_f["Store"].isin(stores)].groupby("Store")[f"POST_{result_val}"].mean().reindex(stores)


pre_mean = df[df["Store"].isin(plot_stores)].groupby("Store")[result_val].mean().reindex(plot_stores)
post_conv = get_plot_data(df_conv, plot_stores)
post_static = get_plot_data(df_static, plot_stores)
post_gnn = get_plot_data(df_gnn, plot_stores)

x = np.arange(len(plot_stores))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - 1.5 * width, pre_mean, width, label="PRE", color="gray", alpha=0.5)
plt.bar(x - 0.5 * width, post_conv, width, label="DoWhy", color="tab:blue")
plt.bar(x + 0.5 * width, post_static, width, label="Oiwa", color="tab:orange")
plt.bar(x + 1.5 * width, post_gnn, width, label="GNN", color="tab:red")

plt.title("Rossmann Sales Prediction")
plt.xticks(x, plot_stores, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("exp_result/rossmann_comparison.png")

print("\n************************\n実行時間の比較\n************************")
print(f"大岩手法 (Static): {end_static - start_static:.4f} sec")
print(f"改善手法 (GNN): Total {end_gnn_whatif - start_gnn_train:.4f} sec")
print(f"Graph saved to exp_result/rossmann_comparison.png")
