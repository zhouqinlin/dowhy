import numpy as np, pandas as pd
import json
import networkx as nx
from dowhy import gcm
from causal_query import CausalQuery
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ttest_rel
# from verify import verify_extended_dataset # 必要に応じて有効化
import time

# 結果検証のためのt検定 attrとPOST_attrを比較
def paired_ttest(df, attr):
    # サンプルデータフレーム（df）は既に存在すると仮定
    # 各 Store ごとに対応のある t 検定を実施
    results = []
    stores = df['Store'].unique()  # ユニークな Store 名を取得

    for store in stores:
        # 各 store のデータを取得
        store_data = df[df['Store'] == store]
        
        # データ数が少なすぎる場合はスキップ
        if len(store_data) < 2:
            continue

        # 対応のある t 検定を実行
        t_stat, p_value = ttest_rel(store_data[attr], store_data[f'POST_{attr}'])
        
        # 結果を保存
        results.append({'Store': store, 't_stat': t_stat, 'p_value': p_value})

    # 検定結果をデータフレームに変換
    results_df = pd.DataFrame(results)

    # 有意水準 0.05 以下の結果を表示
    if not results_df.empty:
        significant_results = results_df[results_df['p_value'] < 0.05]
        print("\n有意な差が検出された Store:")
        print(significant_results)
    else:
        print("\n有意な差は検出されませんでした（データ不足の可能性あり）")


# データセットの読み込み (Rossmann Store Sales)
# データ形式: "Store","DayOfWeek","Date","Sales","Customers","Open","Promo","StateHoliday","SchoolHoliday"
df = pd.read_csv("datasets/rossmann_store_sales.csv", low_memory=False)

# 前処理: 開店しており、売上が発生している行のみ対象
df = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()

# データ量の調整
# ここではデモ用に直近のデータを抽出（Date列がある場合）
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    # 例として2015年7月のデータを抽出
    df = df[df['Date'].dt.strftime('%Y') == '2014']  # Commented out to include all data
    # 文字列に戻す（カテゴリ変数として扱うため）
    df['Date'] = df['Date'].astype(str)

# 必要なカラムの選択
target_cols = ['Store', 'Date', 'DayOfWeek', 'Sales', 'Customers', 'Promo']
df = df[target_cols].dropna()

# 型変換
df['Store'] = df['Store'].astype(str) # 識別子として扱う
df['Promo'] = df['Promo'].astype(int)
df['Sales'] = df['Sales'].astype(float)

print(df.head())
print(f"Data Shape: {df.shape}")


# 分析対象の設定
target_day = 5 # 金曜日 (Amazonでのカテゴリ選択に相当)
print(f"Intervention Target DayOfWeek: {target_day} (Friday)")


# 介入条件
# Calculate average sales for each store and select the top store
store_avg_sales = df.groupby('Store')['Sales'].mean()
top_store = store_avg_sales.idxmax()

# Filter data to include only the top store
top_store_data = df[df['Store'] == top_store]

# Modify interventions to stop promo on Fridays for the top store
interventions = {
    "Promo": {
        # 特定の曜日(DayOfWeek=5)に対して介入
        "condition": lambda row: row["DayOfWeek"] == target_day and row["Store"] == top_store, 
        "intervention": lambda x: 0 # プロモーションを停止する (1 -> 0)
    }
}
result_val = 'Sales'


# 従来のWhat-If問合せ
# DAG定義: Promo, DayOfWeek, Customers が Sales に影響する単純な構造
causal_graph = nx.DiGraph([
    ('DayOfWeek', 'Sales'),
    ('Promo', 'Sales'),
    ('Customers', 'Sales'),
    ('Promo', 'Customers'), # Promoは客数にも影響
    ('DayOfWeek', 'Customers'),
    ('Store', 'Sales') # 店舗ごとのベースライン
])

causal_model = gcm.ProbabilisticCausalModel(causal_graph)
gcm.auto.assign_causal_mechanisms(causal_model, df)
gcm.fit(causal_model, df)

# convresult = gcm.interventional_samples(causal_model, {'price': lambda x: x*0.5}, observed_data=df)
convresult = gcm.interventional_samples(causal_model, interventions, observed_data=df)
# ターゲット曜日のみ抽出
convresult = convresult.loc[convresult['DayOfWeek'] == target_day]

print("Conventional Result Head:")
print(convresult.head())


# データの絞り込み
# Amazonコードと同様のフィルタリングロジック（ここではStoreごとのデータ数を確認）
group_counts = convresult.groupby("Store").size()
# データ数が一定以上のStoreを対象にする（期間が短い場合は閾値を下げる）
valid_stores = group_counts[group_counts >= 1].index 
filtered_conv = convresult[convresult["Store"].isin(valid_stores)]

print(f"従来手法に対するt検定")
paired_ttest(filtered_conv, result_val)

groupby_convresult = filtered_conv.groupby(['Store'])[['Promo', 'POST_Promo', 'Sales', 'POST_Sales']].mean()


# 提案手法によるWhat-If問合せ
start_all = time.time()
causal_query = CausalQuery()
agg_func = 'mean'
groupby_col = 'Date' # 同じ「日付」の他の店舗のPromo状況を集約（競合干渉）

# DAG設定: 他店のPromo平均(Date単位の集約)が自店のSalesに影響を与えると仮定
causal_query.set_causal_graph(
    [
        ('DayOfWeek', 'Sales'),
        ('Promo', 'Sales'),
        ('Customers', 'Sales'),
        ('Promo', 'Customers'),
        ('DayOfWeek', 'Customers'),
        ('Store', 'Sales')
    ],
    [('Promo', 'Sales')], # 拡張エッジ（他店のPromo -> 自店のSales）
    groupby_col, 
    agg_func
)

start_exdata = time.time()
# blockcol='Store' により、"同じDateだが異なるStore" のPromoを集約
ex_training_data = causal_query.extend_dataset(df, blockcol='Store')
end_exdata = time.time()

# 拡張データの確認
# print(ex_training_data.head())

start_train = time.time()
causal_query.train_causal_model(df, ex_training_data)
end_train = time.time()

start_whatif = time.time()
proresult = causal_query.what_if(ex_training_data, interventions)
end_all = time.time()

proresult = proresult[proresult['DayOfWeek'] == target_day]

# データの絞り込み
filtered_pro = proresult[proresult["Store"].isin(valid_stores)]
print(f"提案手法に対するt検定")
paired_ttest(filtered_pro, result_val)

groupby_proresult = filtered_pro.groupby(['Store'])[['Promo', 'POST_Promo', 'Sales', 'POST_Sales']].mean()


# グラフ作成
# フォントサイズを設定
base_font_size = 10 
rcParams.update({
    'font.size': base_font_size,
    'axes.titlesize': base_font_size * 1.2,
    'axes.labelsize': base_font_size,
    'xtick.labelsize': base_font_size * 0.9,
    'ytick.labelsize': base_font_size * 0.8,
    'legend.fontsize': base_font_size * 0.7
})

# 売上規模が大きい上位10店舗を表示
top_stores = df.groupby('Store')['Sales'].mean().sort_values(ascending=False).head(10).index
groupby_proresult_plot = groupby_proresult.loc[top_stores]
groupby_convresult_plot = groupby_convresult.loc[top_stores]

stores_label = groupby_proresult_plot.index # 店舗名
x = np.arange(len(stores_label))
pre_result = groupby_proresult_plot[result_val]  # 平均売上
post_convresult = groupby_convresult_plot[f'POST_{result_val}']
post_proresult = groupby_proresult_plot[f'POST_{result_val}']  # POST平均売上

# 棒グラフの幅と位置設定
bar_width = 0.2
offset = bar_width

# グラフの描画
plt.figure(figsize=(10, 6))
plt.bar(x, pre_result, width=bar_width, label=f'PRE {result_val}', color='tab:blue')
plt.bar(x+offset, post_convresult, width=bar_width, label=f'POST {result_val} (DoWhy)', color='tab:green')
plt.bar(x+2*offset, post_proresult, width=bar_width, label=f'POST {result_val} (Proposed Method)', color='tab:orange')

# グラフの装飾
plt.title(f"Impact of Promo Intervention on {result_val} by Store")
plt.xlabel("Store")
plt.xticks(x + bar_width, stores_label, rotation=45, ha='right')
plt.ylabel(f"{result_val}")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# グラフを保存
plt.savefig(f"exp_result/{result_val}_rossmann_ex({agg_func}).png", dpi=300)
plt.savefig(f"exp_result/{result_val}_rossmann_ex({agg_func}).pdf", dpi=300)

plt.close()


# 結果の表示
print(f"更新前：\n{pre_result.head(10)}")
print(f"更新後（DoWhy）：\n{post_convresult.head(10)}")
print(f"更新後（提案手法）：\n{post_proresult.head(10)}")

# 実行時間の表示
print(f"************************\n提案手法の実行時間\n************************\n")
print(f"データ量：{len(ex_training_data)}")
print(f"データセット拡張に要した時間：{end_exdata-start_exdata}")
print(f"モデルの学習に要した時間：{end_train-start_train}")
print(f"What-If分析に要した時間：{end_all-start_whatif}")
print(f"全体の実行時間：{end_all-start_all}")
