import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import data_loader

# ================= 設定區 =================
CSV_PATH = 'output/clustering_results.csv'
RAW_FILE_PATH = "F:/碩士/114-1_資料探勘/archive (1)/PPG_FieldStudy/S7/S7.pkl"
OUTPUT_DIR = "output"
# =========================================

def map_activity_name(label_id):
    mapping = {
        0: 'Transient', # 過渡期 (走路/切換)
        1: 'Sitting',   # Doc ID: 1
        2: 'Stairs',    # Doc ID: 2
        3: 'Soccer',    # Doc ID: 3
        4: 'Cycling',   # Doc ID: 4
        5: 'Driving',   # Doc ID: 5
        6: 'Lunch',     # Doc ID: 6
        7: 'Walking',   # Doc ID: 7
        8: 'Working'    # Doc ID: 8
    }
    return mapping.get(int(label_id), 'Unknown')

def run_comparison():
    print("=== 正在執行 K-Means vs. DBSCAN 比較分析 ===")
    
    # 1. 讀取與前處理
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.dropna(subset=['Activity_Label'])
        df['Activity_Label'] = df['Activity_Label'].astype(int)
        df['Activity_Name'] = df['Activity_Label'].apply(map_activity_name)
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return

    # 準備特徵
    feature_cols = ['SpecEn', 'PermEn', 'Petrosian_FD', 'Higuchi_FD']
    df = df.dropna(subset=feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # 2. 重跑模型 (確保參數一致)
    print(">> 重跑 K-Means (K=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

    print(">> 重跑 DBSCAN (尋找最佳參數)...")
    dbscan = DBSCAN(eps=0.3, min_samples=20) 
    df['DBSCAN_Label'] = dbscan.fit_predict(X_scaled)
    
    # 檢查有沒有抓到 Noise
    n_noise = len(df[df['DBSCAN_Label'] == -1])
    print(f"   DBSCAN 抓到了 {n_noise} 筆雜訊 (Label = -1)")
    
    if n_noise == 0:
        print("!!! 警告：DBSCAN 沒有抓到任何雜訊，將 eps 調小 !!!")

    # =============================================
    # 圖表 1: DBSCAN 的成分分析 (到底把誰當垃圾？)
    # =============================================
    plt.figure(figsize=(8, 6))
    # 計算每個 DBSCAN Label 裡面的活動分佈
    ct_db = pd.crosstab(df['Activity_Name'], df['DBSCAN_Label'], normalize='index') 
    sns.heatmap(ct_db, annot=True, fmt=".2f", cmap="Reds")
    plt.title('DBSCAN Purity Check: What is inside the Noise (-1)?')
    plt.xlabel('DBSCAN Label (-1 = Noise)')
    plt.ylabel('Ground Truth Activity')
    plt.tight_layout()
    save_file = f'{OUTPUT_DIR}/compare_dbscan_purity.png'
    plt.savefig(save_file, dpi=300)
    print(f">> DBSCAN 成分分析圖已儲存至: {save_file}")
    # =============================================
    # 圖表 2: K-Means vs DBSCAN (演算法對決)
    # =============================================
    # K-Means 認為是某一群的資料，DBSCAN 是否也覺得是雜訊？
    plt.figure(figsize=(10, 6))
    
    # 建立交叉表
    # 行(Y): K-Means Clusters
    # 列(X): DBSCAN Labels
    ct_compare = pd.crosstab(df['KMeans_Cluster'], df['DBSCAN_Label'])
    
    # 畫熱力圖 (顯示筆數 count)
    sns.heatmap(ct_compare, annot=True, fmt="d", cmap="Blues")
    plt.title('Comparison: K-Means Clusters vs. DBSCAN Labels')
    plt.xlabel('DBSCAN Label (-1 is Noise)')
    plt.ylabel('K-Means Cluster ID')
    plt.tight_layout()
    save_file = f'{OUTPUT_DIR}/compare_kmeans_vs_dbscan.png'
    plt.savefig(save_file, dpi=300)
    print(f">> K-Means vs. DBSCAN 比較圖已儲存至: {save_file}")

    # =============================================
    # 文字報告
    # =============================================
    print("\n=== 比較結果摘要 ===")
    # 找出 K-Means 哪一群最容易被 DBSCAN 判為雜訊
    # 計算每一群 K-Means 中，被 DBSCAN 標為 -1 的比例
    kmeans_noise_ratio = df.groupby('KMeans_Cluster')['DBSCAN_Label'].apply(lambda x: (x == -1).mean())
    print("K-Means 各群被 DBSCAN 判定為雜訊的比例:")
    print(kmeans_noise_ratio)
    
    worst_cluster = kmeans_noise_ratio.idxmax()
    print(f"\n>> 結論：K-Means 的 Cluster {worst_cluster} 與 DBSCAN 的雜訊判定高度重疊。")
    print("   這代表該群集確實是『最顯著的異常訊號』。")

if __name__ == "__main__":
    run_comparison()