import numpy as np, pandas as pd
import json
import networkx as nx
from dowhy import gcm
from causal_query import CausalQuery
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ttest_rel
from verify import verify_extended_dataset
import time


# 結果検証のためのt検定 attrとPOST_attrを比較
def paired_ttest(df, attr):
    # サンプルデータフレーム（df）は既に存在すると仮定
    # 各 store ごとに対応のある t 検定を実施
    results = []
    stores = df["store"].unique()  # ユニークな store 名を取得

    for store in stores:
        # 各 store のデータを取得
        store_data = df[df["store"] == store]

        # 対応のある t 検定を実行
        t_stat, p_value = ttest_rel(store_data[attr], store_data[f"POST_{attr}"])

        # 結果を保存
        results.append({"store": store, "t_stat": t_stat, "p_value": p_value})

    # 検定結果をデータフレームに変換
    results_df = pd.DataFrame(results)

    # 有意水準 0.05 以下の結果を表示
    significant_results = results_df[results_df["p_value"] < 0.05]
    print("\n有意な差が検出された store:")
    print(significant_results)


df = pd.read_csv("datasets/products_categories_67k.csv", index_col=0)
df = df[["categories", "average_rating", "price", "store"]]
print(df)
# print(df.shape)
# print(df.groupby(['categories', 'store'])['average_rating'].mean())

# correlation = df['price'].corr(df['rating_number'])
# print("price と rating_numberの相関係数(Pearson)")
# print(f"全体: {correlation:.2f}")
# temp_df = df.query('categories == "Computers"')
# correlation = temp_df['price'].corr(temp_df['rating_number'])
# print(f"categories = 'Computers': {correlation:.2f}")


# 介入条件
interventions = {
    "price": {
        "condition": lambda row: row["store"] == "Intel" and row["categories"] == "Computers & Accessories",
        # "condition": lambda row: row["store"] == 'Intel' and row['main_category'] == 'Computers',
        # "condition": lambda row: True,
        "intervention": lambda x: x * 0.5,
    }
}
result_val = "average_rating"


# 従来のWhat-If問合せ
causal_model = gcm.ProbabilisticCausalModel(
    nx.DiGraph(
        [
            ("store", "categories"),
            ("store", "price"),
            ("categories", "price"),
            ("price", "average_rating"),
            ("store", "average_rating"),
        ]
    )
)
gcm.auto.assign_causal_mechanisms(causal_model, df)
gcm.fit(causal_model, df)

# convresult = gcm.interventional_samples(causal_model, {'price': lambda x: x*0.5}, observed_data=df)
convresult = gcm.interventional_samples(causal_model, interventions, observed_data=df)
convresult = convresult.loc[convresult["categories"] == "Computers & Accessories"]
# convresult[f"POST_{result_val}"] = convresult[f"{result_val}"]
# convresult[f"{result_val}"] = df[f"{result_val}"]
# temp = convresult.query('store == "Sony"')
# print(f"conventional result:\n{temp}")
print(convresult)
# print(f"従来手法に対するt検定")
# paired_ttest(convresult, result_val)


# データの絞り込み
# グループの要素数を計算
group_counts = convresult.groupby("store").size()
# 要素数が1000未満のグループをフィルタリング
valid_store = group_counts[group_counts >= 1000].index
# 条件を満たす行のみ残す
filtered_conv = convresult[convresult["store"].isin(valid_store)]
print(f"従来手法に対するt検定")
paired_ttest(filtered_conv, result_val)
# print(filtered_conv.groupby("store").size())
groupby_convresult = filtered_conv.groupby(["store"])[
    ["price", "POST_price", "average_rating", "POST_average_rating"]
].mean()

# # グラフ作成
# x = groupby_convresult.index.get_level_values('store')  # 店舗名
# pre_result = groupby_convresult[result_val]  # 平均評価
# post_result = groupby_convresult[f'POST_{result_val}']  # POST平均評価

# width = 0.4  # 棒グラフの幅

# # グラフの描画
# plt.figure(figsize=(10, 8))
# plt.bar(x, pre_result, width=width, label=f'{result_val}', align='center')
# plt.bar(x, post_result, width=width, label=f'POST {result_val}', align='edge')

# # グラフの装飾
# plt.title(f"{result_val} vs POST {result_val} for 'Computers & Accessories'")
# plt.xlabel("Store")
# plt.xticks(rotation=45, ha='right')  # 45度回転、右揃え
# plt.ylabel(f"{result_val}")
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()

# # グラフを保存
# plt.savefig(f"exp_result/{result_val}_computers.png", dpi=300)
# plt.savefig(f"exp_result/{result_val}_computers.pdf", dpi=300)

# # 表示を省略する場合（サーバー環境向け）
# plt.close()


# 提案手法によるWhat-If問合せ
start_all = time.time()
causal_query = CausalQuery()
agg_func = "mean"
groupby_col = "categories"
causal_query.set_causal_graph(
    [("store", "categories"), ("store", "price"), ("price", "average_rating"), ("store", "average_rating")],
    [("price", "average_rating")],
    groupby_col,
    agg_func,
)

start_exdata = time.time()
ex_training_data = causal_query.extend_dataset(df, blockcol="store")
end_exdata = time.time()
# print(ex_training_data)
# verify_extended_dataset(df, ex_training_data, groupby_col, 'price', 'mean', 'store')

start_train = time.time()
causal_query.train_causal_model(df, ex_training_data)
end_train = time.time()

start_whatif = time.time()
proresult = causal_query.what_if(ex_training_data, interventions)
end_all = time.time()

proresult = proresult[proresult["categories"] == "Computers & Accessories"]
# print(f"提案手法に対するt検定")
# paired_ttest(proresult, result_val)
# print(proresult)

# データの絞り込み
# 条件を満たす行のみ残す
filtered_pro = proresult[proresult["store"].isin(valid_store)]
print(f"提案手法に対するt検定")
paired_ttest(filtered_pro, result_val)

groupby_proresult = filtered_pro.groupby(["categories", "store"])[
    ["price", "POST_price", "average_rating", "POST_average_rating"]
].mean()
# groupby_proresult.to_csv("datasets/result_products.csv", index=True)

# グラフ作成
# フォントサイズを設定（横幅に応じて調整）
base_font_size = 10  # 基本の文字サイズ
rcParams.update(
    {
        "font.size": base_font_size,  # 全体のフォントサイズ
        "axes.titlesize": base_font_size * 1.2,  # タイトル
        "axes.labelsize": base_font_size,  # 軸ラベル
        "xtick.labelsize": base_font_size * 0.9,  # X軸目盛り
        "ytick.labelsize": base_font_size * 0.8,  # Y軸目盛り
        "legend.fontsize": base_font_size * 0.7,  # 凡例
    }
)
stores = groupby_proresult.index.get_level_values("store")  # 店舗名
x = np.arange(len(groupby_proresult.index))
pre_result = groupby_proresult[result_val]  # 平均評価
post_convresult = groupby_convresult[f"POST_{result_val}"]
post_proresult = groupby_proresult[f"POST_{result_val}"]  # POST平均評価
# 棒グラフの幅と位置設定
bar_width = 0.2  # 各棒の幅
offset = bar_width  # 棒を横にずらす量

# グラフの描画
plt.figure(figsize=(5.9, 5))
plt.bar(x, pre_result, width=bar_width, label=f"PRE {result_val}", color="tab:blue")
plt.bar(x + offset, post_convresult, width=bar_width, label=f"POST {result_val} (DoWhy)", color="tab:green")
plt.bar(
    x + 2 * offset, post_proresult, width=bar_width, label=f"POST {result_val} (Proposed Method)", color="tab:orange"
)

# グラフの装飾
plt.title(f"")
plt.xlabel("Store")
plt.xticks(x + bar_width, stores, rotation=45, ha="right")  # 45度回転、右揃え
plt.ylabel(f"{result_val}")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# グラフを保存
plt.savefig(f"exp_result/{result_val}_computers_ex({agg_func}).png", dpi=300)
plt.savefig(f"exp_result/{result_val}_computers_ex({agg_func}).pdf", dpi=300)

# 表示を省略する場合（サーバー環境向け）
plt.close()


# 結果の表示
print(f"更新前：\n{pre_result}")
print(f"更新後（DoWhy）：\n{post_convresult}")
print(f"更新後（提案手法）：\n{post_proresult}")

# 実行時間の表示
print(f"************************\n提案手法の実行時間\n************************\n")
print(f"データ量：{len(ex_training_data)}")
print(f"データセット拡張に要した時間：{end_exdata-start_exdata}")
print(f"モデルの学習に要した時間：{end_train-start_train}")
print(f"What-If分析に要した時間：{end_all-start_whatif}")
print(f"全体の実行時間：{end_all-start_all}")
