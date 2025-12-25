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
    # 各 neighbourhood_cleansed ごとに対応のある t 検定を実施
    results = []
    neighbourhoods = df['neighbourhood_cleansed'].unique()  # ユニークな neighbourhood 名を取得

    for neighbourhood in neighbourhoods:
        # 各 neighbourhood のデータを取得
        neighbourhood_data = df[df['neighbourhood_cleansed'] == neighbourhood]
        
        # データ数が少なすぎる場合はスキップ
        if len(neighbourhood_data) < 2:
            continue

        # 対応のある t 検定を実行
        t_stat, p_value = ttest_rel(neighbourhood_data[attr], neighbourhood_data[f'POST_{attr}'])
        
        # 結果を保存
        results.append({'neighbourhood_cleansed': neighbourhood, 't_stat': t_stat, 'p_value': p_value})

    # 検定結果をデータフレームに変換
    results_df = pd.DataFrame(results)

    # 有意水準 0.05 以下の結果を表示
    if not results_df.empty:
        significant_results = results_df[results_df['p_value'] < 0.05]
        print("\n有意な差が検出された neighbourhood_cleansed:")
        print(significant_results)
    else:
        print("\n有意な差は検出されませんでした（データ不足の可能性あり）")


# データセットの読み込み (Airbnb Listings Data)
df = pd.read_csv("datasets/airbnb_cleaned.csv")

# 必要なカラムの選択とリネーム
# neighbourhood_cleansed (グループ/介入単位)
# room_type (カテゴリ/層別化単位)
target_cols = ['room_type', 'review_scores_rating', 'price', 'neighbourhood_cleansed']
df = df[target_cols].dropna()

# Airbnbデータの価格は "$1,200.00" のような文字列の可能性があるため数値変換
if df['price'].dtype == 'O':
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# レビュースコアが欠損している行や、異常値を除外
df = df[df['review_scores_rating'] > 0]

print(df.head())
print(f"Data Shape: {df.shape}")


# 分析対象の設定
target_room_type = 'Entire home/apt'
# 介入対象のネイバーフッドを動的に決定（最も物件数が多い地域をターゲットにする）
target_neighbourhood = df[df['room_type'] == target_room_type]['neighbourhood_cleansed'].value_counts().idxmax()
print(f"Intervention Target Neighbourhood: {target_neighbourhood}")


# 介入条件
interventions = {
    "price": {
        # 特定の地域(neighbourhood)かつ特定の部屋タイプ(room_type)に対して介入
        "condition": lambda row: row["neighbourhood_cleansed"] == target_neighbourhood and row['room_type'] == target_room_type, 
        "intervention": lambda x: x * 0.85 # 価格を15%下げる介入
    }
}
result_val = 'review_scores_rating'


# 従来のWhat-If問合せ
# DAG定義: neighbourhoodとroom_typeが価格と評価に影響し、価格が評価に影響する
causal_graph = nx.DiGraph([
    ('neighbourhood_cleansed', 'room_type'),
    ('neighbourhood_cleansed', 'price'),
    ('room_type', 'price'),
    ('price', 'review_scores_rating'),
    ('neighbourhood_cleansed', 'review_scores_rating')
])

causal_model = gcm.ProbabilisticCausalModel(causal_graph)
gcm.auto.assign_causal_mechanisms(causal_model, df)
gcm.fit(causal_model, df)

# convresult = gcm.interventional_samples(causal_model, {'price': lambda x: x*0.5}, observed_data=df)
convresult = gcm.interventional_samples(causal_model, interventions, observed_data=df)
convresult = convresult.loc[convresult['room_type'] == target_room_type]

print("Conventional Result Head:")
print(convresult.head())


# データの絞り込み
# グループの要素数を計算
group_counts = convresult.groupby("neighbourhood_cleansed").size()
# 要素数が一定数（例:100）以上のグループをフィルタリング（AirbnbはAmazonより粒度が細かい場合があるため閾値を調整）
valid_neighbourhood = group_counts[group_counts >= 100].index
# 条件を満たす行のみ残す
filtered_conv = convresult[convresult["neighbourhood_cleansed"].isin(valid_neighbourhood)]

print(f"従来手法に対するt検定")
paired_ttest(filtered_conv, result_val)

groupby_convresult = filtered_conv.groupby(['neighbourhood_cleansed'])[['price', 'POST_price', 'review_scores_rating', 'POST_review_scores_rating']].mean()


# 提案手法によるWhat-If問合せ
start_all = time.time()
causal_query = CausalQuery()
agg_func = 'mean'
groupby_col = 'room_type' # カテゴリ単位で集約（同じ部屋タイプ内の他地域の価格平均）

# DAG設定: 近隣相場(Average Price of other neighbourhoods)が評価に影響を与えると仮定
causal_query.set_causal_graph(
    [
        ('neighbourhood_cleansed', 'room_type'),
        ('neighbourhood_cleansed', 'price'),
        ('price', 'review_scores_rating'),
        ('neighbourhood_cleansed', 'review_scores_rating')
    ],
    [('price', 'review_scores_rating')], # 拡張エッジ（他地域の価格が評価に干渉）
    groupby_col, 
    agg_func
)

start_exdata = time.time()
# blockcol='neighbourhood_cleansed' により、"同じRoom Typeだが異なるNeighbourhood" の価格を集約
ex_training_data = causal_query.extend_dataset(df, blockcol='neighbourhood_cleansed')
end_exdata = time.time()

# 拡張データの確認（デバッグ用）
# print(ex_training_data.head())

start_train = time.time()
causal_query.train_causal_model(df, ex_training_data)
end_train = time.time()

start_whatif = time.time()
proresult = causal_query.what_if(ex_training_data, interventions)
end_all = time.time()

proresult = proresult[proresult['room_type'] == target_room_type]

# データの絞り込み
# 条件を満たす行のみ残す
filtered_pro = proresult[proresult["neighbourhood_cleansed"].isin(valid_neighbourhood)]
print(f"提案手法に対するt検定")
paired_ttest(filtered_pro, result_val)

groupby_proresult = filtered_pro.groupby(['room_type', 'neighbourhood_cleansed'])[['price', 'POST_price', 'review_scores_rating', 'POST_review_scores_rating']].mean()


# グラフ作成
# フォントサイズを設定（横幅に応じて調整）
base_font_size = 10  # 基本の文字サイズ
rcParams.update({
    'font.size': base_font_size,       # 全体のフォントサイズ
    'axes.titlesize': base_font_size * 1.2,  # タイトル
    'axes.labelsize': base_font_size,        # 軸ラベル
    'xtick.labelsize': base_font_size * 0.9, # X軸目盛り
    'ytick.labelsize': base_font_size * 0.8, # Y軸目盛り
    'legend.fontsize': base_font_size * 0.7  # 凡例
})

# 物件数(データサイズ)が多い上位10エリアを選定
top_n = 15
# 元データdfでneighbourhoodごとの件数を数え、多い順にindexを取得
top_neighbourhoods = df['neighbourhood_cleansed'].value_counts().head(top_n).index
groupby_proresult_plot = groupby_proresult[groupby_proresult.index.get_level_values('neighbourhood_cleansed').isin(top_neighbourhoods)]
groupby_convresult_plot = groupby_convresult[groupby_convresult.index.isin(top_neighbourhoods)]

neighbourhoods = groupby_proresult_plot.index.get_level_values('neighbourhood_cleansed')  # 店舗名(地域名)
x = np.arange(len(neighbourhoods))
pre_result = groupby_proresult_plot[result_val]  # 平均評価
post_convresult = groupby_convresult_plot[f'POST_{result_val}']
post_proresult = groupby_proresult_plot[f'POST_{result_val}']  # POST平均評価

# 棒グラフの幅と位置設定
bar_width = 0.2  # 各棒の幅
offset = bar_width  # 棒を横にずらす量

# グラフの描画
plt.figure(figsize=(10, 6)) # サイズ調整
plt.bar(x, pre_result, width=bar_width, label=f'PRE {result_val}', color='tab:blue')
plt.bar(x+offset, post_convresult, width=bar_width, label=f'POST {result_val} (DoWhy)', color='tab:green')
plt.bar(x+2*offset, post_proresult, width=bar_width, label=f'POST {result_val} (Proposed Method)', color='tab:orange')

# グラフの装飾
plt.title(f"Impact of Price Intervention on {result_val} by Neighbourhood")
plt.xlabel("Neighbourhood")
plt.xticks(x + bar_width, neighbourhoods, rotation=45, ha='right')  # 45度回転、右揃え
plt.ylabel(f"{result_val}")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# グラフを保存
plt.savefig(f"exp_result/{result_val}_airbnb_ex({agg_func}).png", dpi=300)
plt.savefig(f"exp_result/{result_val}_airbnb_ex({agg_func}).pdf", dpi=300)

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
