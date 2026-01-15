import pandas as pd
import os

# ==========================================
# 設定
# ==========================================
RESULTS_DIR = "exp_result"
OUTPUT_FILE = f"{RESULTS_DIR}/rossmann_lift_for_excel.csv"

# 実験の定義 (ファイル名のプレフィックス: Excelでの表示名)
EXPERIMENTS = {
    "rossmann_promo0": "Stop Promo (0)",
    "rossmann_promo1": "Start Promo (1)",
    "rossmann_do_nothing": "Do Nothing"
}

def calc_mean_lift(prefix):
    """各実験の平均Lift (POST - PRE) を計算する"""
    prop_path = f"{RESULTS_DIR}/{prefix}_proposed_method_aggregated.csv"
    conv_path = f"{RESULTS_DIR}/{prefix}_conventional_method_aggregated.csv"
    
    # ファイルが存在しない場合はNoneを返す
    if not os.path.exists(prop_path) or not os.path.exists(conv_path):
        print(f"Warning: Files for {prefix} not found.")
        return None, None

    df_prop = pd.read_csv(prop_path)
    df_conv = pd.read_csv(conv_path)

    # Lift = POST_Sales - Sales
    lift_prop = (df_prop['POST_Sales'] - df_prop['Sales']).mean()
    lift_conv = (df_conv['POST_Sales'] - df_conv['Sales']).mean()
    
    return lift_prop, lift_conv

# ==========================================
# データ集計
# ==========================================
data_for_excel = []

print("集計を開始します...")

for prefix, label in EXPERIMENTS.items():
    l_prop, l_conv = calc_mean_lift(prefix)
    
    if l_prop is not None:
        data_for_excel.append({
            "Experiment": label,
            "Proposed Method": l_prop,
            "Conventional (DoWhy)": l_conv
        })

# DataFrame化
df = pd.DataFrame(data_for_excel)

# Excelで読み込みやすいように列の順序を整理
df = df[["Experiment", "Proposed Method", "Conventional (DoWhy)"]]

# CSV保存 (Excelで文字化けしないよう utf-8-sig を使用)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\n完了しました。以下のファイルをExcelで開いてください:")
print(f" -> {OUTPUT_FILE}")
print("\nデータ内容:")
print(df)