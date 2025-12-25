import numpy as np, pandas as pd
import json
import networkx as nx
from dowhy import gcm
from causal_query import CausalQuery
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ttest_rel

# from verify import verify_extended_dataset # verify is not provided, commented out
import time

# from causaldata import abortion # Assuming you can import this, or load csv


# 結果検証のためのt検定 attrとPOST_attrを比較
def paired_ttest(df, attr, group_col):
    # サンプルデータフレーム（df）は既に存在すると仮定
    # 各 group_col (e.g., year) ごとに対応のある t 検定を実施
    results = []
    groups = df[group_col].unique()  # ユニークな group 名を取得

    for group in groups:
        # 各 group のデータを取得
        group_data = df[df[group_col] == group]

        # データが十分にある場合のみ実行
        if len(group_data) > 1:
            # 対応のある t 検定を実行
            t_stat, p_value = ttest_rel(group_data[attr], group_data[f"POST_{attr}"])
            # 結果を保存
            results.append({group_col: group, "t_stat": t_stat, "p_value": p_value})

    # 検定結果をデータフレームに変換
    results_df = pd.DataFrame(results)

    # 有意水準 0.05 以下の結果を表示
    if not results_df.empty:
        significant_results = results_df[results_df["p_value"] < 0.05]
        print(f"\n有意な差が検出された {group_col}:")
        print(significant_results)
    else:
        print(f"\n{group_col} ごとの検定結果はありません（データ不足の可能性）")


# データ読み込み (causaldata abortion dataset)
df = pd.read_csv("datasets/abortion.csv")

columns_to_keep = ["fip", "year", "repeal", "lnr", "poverty"]
df = df[columns_to_keep]

df.dropna(inplace=True)

# データ型の調整
# fip (State ID) は数値ではなくカテゴリとして扱うべき
df["fip"] = df["fip"].astype(str)


print(df.head())
print(f"Data Shape: {df.shape}")

# 設定
groupby_col = "fip"  # 状態 (State) - 元の categories に相当
block_col = "year"  # 年 (Year) - 元の store に相当
treatment = "repeal"  # 介入変数
outcome = "lnr"  # 結果変数 (log gonorrhea rate)
result_val = outcome


# 介入条件
# まだ repeal していない州に対して、repeal=1 にする介入
interventions = {
    treatment: {
        "condition": lambda row: row[treatment] == 0,
        "intervention": lambda x: 1,
    }
}


# 従来のWhat-If問合せ (Simple DAG)
# Graph: State/Year -> Repeal -> Outcome, State/Year -> Outcome, Poverty -> Outcome
causal_model = gcm.ProbabilisticCausalModel(
    nx.DiGraph(
        [
            ("fip", "repeal"),
            ("year", "repeal"),
            ("poverty", "lnr"),
            ("fip", "lnr"),
            ("year", "lnr"),
            ("repeal", "lnr"),
        ]
    )
)
gcm.auto.assign_causal_mechanisms(causal_model, df)
gcm.fit(causal_model, df)

convresult = gcm.interventional_samples(causal_model, interventions, observed_data=df)
# 結果が見やすいようにフィルタ
# convresult = convresult.loc[convresult["fip"] == 1] # Example filter
print("Conventional Result Head:")
print(convresult)


# データの絞り込み
# グループの要素数を計算 (Yearごとのデータ数チェック)
group_counts = convresult.groupby(block_col).size()
# 要素数が十分あるグループをフィルタリング (データセットに合わせて調整)
valid_block = group_counts[group_counts >= 10].index  # abortionデータは小さいので閾値を調整
filtered_conv = convresult[convresult[block_col].isin(valid_block)]

print(f"従来手法に対するt検定")
paired_ttest(filtered_conv, result_val, block_col)

# 集計 (Year ごとの平均)
groupby_convresult = filtered_conv.groupby([block_col])[
    [treatment, f"POST_{treatment}", outcome, f"POST_{outcome}"]
].mean()


# 提案手法によるWhat-If問合せ
start_all = time.time()
causal_query = CausalQuery()
agg_func = "mean"  # 他の州の平均などの集約

# 因果グラフの設定
# ex_edges: ("repeal", "lnr") -> 他の州の repeal が 自州の lnr に影響する (波及効果/Interference)
causal_query.set_causal_graph(
    edges=[
        ("fip", "repeal"),
        ("year", "repeal"),
        ("poverty", "lnr"),
        ("fip", "lnr"),
        ("year", "lnr"),
        ("repeal", "lnr"),
    ],
    ex_edges=[(treatment, outcome)],  # Inter-state effect: Neighbor's repeal affects my LNR
    groupby_col=groupby_col,  # fip (State) ごとの集約ではなく、Yearブロック内での他fipの集約
    agg_func=agg_func,
)

# Note: CausalQuery.extend_dataset の仕様に合わせてパラメータを渡す
# ここでは「同じ year (block_col)」の中で「他の fip (groupby_col)」の値を集約したい
# extend_dataset(df, blockcol='year') -> year内で他のfipを集約
start_exdata = time.time()
ex_training_data = causal_query.extend_dataset(df, blockcol=block_col)
end_exdata = time.time()

start_train = time.time()
causal_query.train_causal_model(df, ex_training_data)
end_train = time.time()

start_whatif = time.time()
proresult = causal_query.what_if(ex_training_data, interventions)
end_all = time.time()

# proresult = proresult[proresult[groupby_col] == ...] # 必要ならフィルタ

# データの絞り込み
filtered_pro = proresult[proresult[block_col].isin(valid_block)]
print(f"提案手法に対するt検定")
paired_ttest(filtered_pro, result_val, block_col)

groupby_proresult = filtered_pro.groupby([groupby_col, block_col])[
    [treatment, f"POST_{treatment}", outcome, f"POST_{outcome}"]
].mean()
# さらに Year だけで再集約してグラフ用にする
groupby_proresult_agg = filtered_pro.groupby([block_col])[
    [treatment, f"POST_{treatment}", outcome, f"POST_{outcome}"]
].mean()


# グラフ作成
base_font_size = 10
rcParams.update(
    {
        "font.size": base_font_size,
        "axes.titlesize": base_font_size * 1.2,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size * 0.9,
        "ytick.labelsize": base_font_size * 0.8,
        "legend.fontsize": base_font_size * 0.7,
    }
)

blocks = groupby_proresult_agg.index  # Years
x = np.arange(len(blocks))
pre_result = groupby_convresult[result_val]  # Original Outcome
post_convresult = groupby_convresult[f"POST_{result_val}"]  # DoWhy
post_proresult = groupby_proresult_agg[f"POST_{result_val}"]  # Proposed

bar_width = 0.2
offset = bar_width

plt.figure(figsize=(8, 5))
plt.bar(x, pre_result, width=bar_width, label=f"PRE {result_val}", color="tab:blue")
plt.bar(x + offset, post_convresult, width=bar_width, label=f"POST {result_val} (DoWhy)", color="tab:green")
plt.bar(
    x + 2 * offset, post_proresult, width=bar_width, label=f"POST {result_val} (Proposed Method)", color="tab:orange"
)

plt.title(f"Impact of {treatment} on {result_val} by {block_col}")
plt.xlabel(block_col)
plt.xticks(x + bar_width, blocks, rotation=45, ha="right")
plt.ylabel(f"{result_val}")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig(f"exp_result/{result_val}_{block_col}_ex({agg_func}).png", dpi=300)
# plt.close()

# 結果の表示
print(f"更新前 (Mean by Year)：\n{pre_result}")
print(f"更新後 DoWhy (Mean by Year)：\n{post_convresult}")
print(f"更新後 提案手法 (Mean by Year)：\n{post_proresult}")

# 実行時間の表示
print(f"************************\n提案手法の実行時間\n************************\n")
print(f"データ量：{len(ex_training_data)}")
print(f"データセット拡張に要した時間：{end_exdata-start_exdata}")
print(f"モデルの学習に要した時間：{end_train-start_train}")
print(f"What-If分析に要した時間：{end_all-start_whatif}")
print(f"全体の実行時間：{end_all-start_all}")
