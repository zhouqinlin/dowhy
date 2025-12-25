import pandas as pd
import numpy as np

def clean_airbnb_data(input_file, output_file):
    print(f"Loading {input_file}...")
    
    # 必要なカラムのみを指定して読み込むことでメモリを節約
    # 'neighbourhood_cleansed': エリア（Amazonデータのstoreに相当）
    # 'room_type': 部屋タイプ（Amazonデータのcategoriesに相当）
    # 'price': 価格（介入変数）
    # 'review_scores_rating': 評価スコア（結果変数）
    target_cols = ['id', 'neighbourhood_cleansed', 'room_type', 'price', 'review_scores_rating']
    
    try:
        # encoding='utf-8' で読み込めない場合は 'utf-8-sig' や 'latin1' を試行
        df = pd.read_csv(input_file, usecols=lambda c: c in target_cols, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Trying 'latin1'...")
        df = pd.read_csv(input_file, usecols=lambda c: c in target_cols, encoding='latin1')

    # カラム名が不足している場合のチェック（neighbourhood_cleansedがない場合はneighbourhoodを探す）
    if 'neighbourhood_cleansed' not in df.columns and 'neighbourhood' in df.columns:
        df.rename(columns={'neighbourhood': 'neighbourhood_cleansed'}, inplace=True)

    print(f"Original shape: {df.shape}")

    # 1. 価格のクリーニング ('$1,200.00' -> 1200.0)
    if df['price'].dtype == 'O':
        df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # 2. 欠損値の削除
    # 価格や評価がないデータは分析に使えないため削除
    df.dropna(subset=['price', 'review_scores_rating', 'neighbourhood_cleansed', 'room_type'], inplace=True)

    # 3. データ型の最適化
    df['review_scores_rating'] = pd.to_numeric(df['review_scores_rating'], errors='coerce')
    
    # 異常値の除外（評価が0のものなど）
    df = df[df['review_scores_rating'] > 0]

    print(f"Processed shape: {df.shape}")
    print(df.head())

    # CSVとして保存
    df.to_csv(output_file, index=False, encoding='utf-8-sig') # Excelでも文字化けしないBOM付きUTF-8
    print(f"Saved cleaned data to {output_file}")

if __name__ == "__main__":
    # ここに入力ファイル名を指定してください
    input_csv = "datasets/listings.csv" 
    output_csv = "datasets/airbnb_cleaned.csv"
    
    clean_airbnb_data(input_csv, output_csv)
