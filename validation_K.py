import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
    return mapping.get(label_id, 'Unknown')

def analyze_and_visualize():
    print("=== 開始執行驗證分析 (Validation) ===")
    
    # 1. 讀取資料
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"成功載入 CSV: {df.shape}")
    except FileNotFoundError:
        print(f"錯誤：找不到 {CSV_PATH}")
        return

    # === 除錯：印出 CSV 裡的 Label 到底長怎樣 ===
    print("\n[Debug] 檢查 'Activity_Label' 欄位內容:")
    unique_labels = df['Activity_Label'].unique()
    print(f"CSV 裡的原始標籤值: {unique_labels}")
    
    if df['Activity_Label'].isnull().all():
        print("!!! 警告：所有標籤都是空值 (NaN)，請檢查 main.py 的特徵提取部分 !!!")
        return

    # 2. 讀取原始訊號
    raw_signal, _, _, fs = data_loader.load_ppg_data(RAW_FILE_PATH)
    if raw_signal is None: return

    # 準備資料
    feature_cols = ['SpecEn', 'PermEn', 'Petrosian_FD', 'Higuchi_FD']
    df = df.dropna(subset=feature_cols) # 去除特徵有缺漏的列
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # ==========================
    # 任務 1: 混淆矩陣
    # ==========================
    print("\n>> 正在建立 K={n_clusters}混淆矩陣...")
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # === 強制轉整數再 Mapping ===
    # 填補空值為 -1，並強制轉為整數 (int)，避免 1.0 對不到 1 的問題
    df['Activity_Label'] = df['Activity_Label'].fillna(-1).astype(int)
    
    # 再次檢查轉型後的標籤
    print(f"轉型後的標籤值: {df['Activity_Label'].unique()}")

    df['Activity_Name'] = df['Activity_Label'].apply(map_activity_name)
    
    # 檢查是否還有 Unknown
    if 'Unknown' in df['Activity_Name'].values:
        print("注意：仍有部分標籤顯示為 Unknown，請確認 map_activity_name 的字典對照表是否完整。")

    # 建立交叉表
    conf_matrix = pd.crosstab(df['Activity_Name'], df['Cluster'], normalize='index')

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title('Activity vs. Cluster Distribution (K=3)')
    plt.ylabel('Ground Truth Activity')
    plt.xlabel('Cluster ID')
    plt.tight_layout()
    save_file = f'{OUTPUT_DIR}/K-means_confusion_matrix.png'
    plt.savefig(save_file, dpi=300)
    print(f">> 混淆矩陣已儲存至: {save_file}")

    # ==========================
    # 任務 2: 波形圖
    # ==========================
    print(">> 正在繪製各群代表性波形... (顯示 3 組)")
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 4))
    axes = np.atleast_1d(axes).flatten()

    for cluster_id in range(n_clusters):
        cluster_samples = df[df['Cluster'] == cluster_id]
        if not cluster_samples.empty:
            sample = cluster_samples.sample(1).iloc[0]
            start_idx = int(sample['Start_Index'])
            window_size = 8 * 64 
            wave_segment = raw_signal[start_idx : start_idx + window_size]
            
            axes[cluster_id].plot(wave_segment, color='#2c3e50', linewidth=1.5)
            top_activity = cluster_samples['Activity_Name'].mode()[0]
            axes[cluster_id].set_title(f"Cluster {cluster_id}\n(Dominant: {top_activity})", fontsize=11, fontweight='bold')
            axes[cluster_id].grid(True, alpha=0.3)
        else:
            axes[cluster_id].set_title(f"Cluster {cluster_id} (Empty)")

    plt.suptitle("Representative Waveforms", fontsize=16)
    plt.tight_layout()
    save_file = f'{OUTPUT_DIR}/K-means_waveforms.png'
    plt.savefig(save_file, dpi=300)
    print(f"[v] 波形圖已儲存")
    

if __name__ == "__main__":
    analyze_and_visualize()