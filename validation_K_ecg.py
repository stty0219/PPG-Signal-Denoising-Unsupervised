import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import data_loader 

# ================= 設定區 =================
CSV_PATH = 'output/clustering_results.csv'
RAW_FILE_PATH = "F:/碩士/114-1_資料探勘/archive (1)/PPG_FieldStudy/S7/S7.pkl"
OUTPUT_DIR = "output"
# =========================================

def calculate_ppg_hr(signal_segment, fs=64):
    """ 計算 PPG 主頻 (BPM) """
    f, Pxx = welch(signal_segment, fs, nperseg=512)
    valid_mask = (f >= 0.5) & (f <= 4.0)
    f_valid = f[valid_mask]
    Pxx_valid = Pxx[valid_mask]
    
    if len(Pxx_valid) == 0: return np.nan
    peak_freq = f_valid[np.argmax(Pxx_valid)]
    return peak_freq * 60

def run_kmeans_ecg_validation():
    print("=== 正在執行 K-Means 心率誤差驗證 ===")
    
    # 1. 讀取資料
    try:
        df = pd.read_csv(CSV_PATH)
        ppg_signal, _, ecg_labels, fs = data_loader.load_ppg_data(RAW_FILE_PATH)
        if ecg_labels is None: return
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return

    # 2. 重建 K-Means (K=3)
    print(">> 重建 K-Means 模型 (K=3)...")
    feature_cols = ['SpecEn', 'PermEn', 'Petrosian_FD', 'Higuchi_FD']
    df = df.dropna(subset=feature_cols)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 預期結果: 2=Clean, 0=Minor, 1=Artifacts
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(X_scaled)

    # 3. 計算心率誤差
    print(f">> 計算 {len(df)} 筆資料的心率誤差...")
    ppg_bpms = []
    ecg_bpms = []
    errors = []
    
    for idx, row in df.iterrows():
        start_idx = int(row['Start_Index'])
        segment = ppg_signal[start_idx : start_idx + 8*fs]
        ppg_hr = calculate_ppg_hr(segment, fs)
        label_idx = start_idx // (2 * fs)
        
        if label_idx < len(ecg_labels):
            ecg_hr = ecg_labels[label_idx]
            error = abs(ppg_hr - ecg_hr)
            ppg_bpms.append(ppg_hr)
            ecg_bpms.append(ecg_hr)
            errors.append(error)
        else:
            ppg_bpms.append(np.nan)
            ecg_bpms.append(np.nan)
            errors.append(np.nan)

    df['PPG_HR'] = ppg_bpms
    df['ECG_HR'] = ecg_bpms
    df['HR_Error'] = errors
    df_final = df.dropna(subset=['HR_Error'])
    
    # 4. 統計數據
    print("\n=== K-Means 各群集心率誤差統計 (MAE) ===")
    mae_stats = df_final.groupby('Cluster_ID')['HR_Error'].agg(['mean', 'std', 'count', 'min', 'max'])
    print(mae_stats.round(2))

    # --- 圖表 A: 箱型圖 ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster_ID', y='HR_Error', data=df_final, palette="viridis", showfliers=False)
    plt.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='Acceptable Error (10 BPM)')
    plt.title('K-Means Heart Rate Error Distribution', fontsize=14)
    plt.xlabel('Cluster ID')
    plt.ylabel('Absolute Error (BPM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/kmeans_ecg_boxplot.png', dpi=300)
    print(f"[v] 誤差箱型圖已儲存: {OUTPUT_DIR}/kmeans_ecg_boxplot.png")

    # --- 圖表 B: 散佈圖 ---
    # 依照品質好壞排列：Cluster 2 (Good) -> Cluster 0 (Mid) -> Cluster 1 (Bad)
    print(">> 正在繪製三方散佈圖...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axis_min, axis_max = 40, 180
    
    # --- 圖 1: Cluster 2 (Clean) ---
    data_2 = df_final[df_final['Cluster_ID'] == 2]
    axes[0].scatter(data_2['ECG_HR'], data_2['PPG_HR'], alpha=0.5, color='#2ecc71', label='Cluster 2') # 綠色
    axes[0].plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7)
    axes[0].set_title(f'Cluster 2: Clean Signals\n(Sitting/Static)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('True Heart Rate (ECG)')
    axes[0].set_ylabel('Estimated Heart Rate (PPG)')
    axes[0].set_xlim(axis_min, axis_max)
    axes[0].set_ylim(axis_min, axis_max)
    axes[0].grid(True, alpha=0.3)

    # --- 圖 2: Cluster 0 (Minor Motion) ---
    data_0 = df_final[df_final['Cluster_ID'] == 0]
    axes[1].scatter(data_0['ECG_HR'], data_0['PPG_HR'], alpha=0.5, color='#9b59b6', label='Cluster 0') # 紫色
    axes[1].plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7)
    axes[1].set_title(f'Cluster 0: Minor Motion\n(Lunch/Cycling)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('True Heart Rate (ECG)')
    axes[1].set_xlim(axis_min, axis_max)
    axes[1].set_ylim(axis_min, axis_max)
    axes[1].grid(True, alpha=0.3)
    
    # --- 圖 3: Cluster 1 (Major Artifacts) ---
    data_1 = df_final[df_final['Cluster_ID'] == 1]
    axes[2].scatter(data_1['ECG_HR'], data_1['PPG_HR'], alpha=0.5, color='#e74c3c', label='Cluster 1') # 紅色
    axes[2].plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7)
    axes[2].set_title(f'Cluster 1: Major Artifacts\n(Walking/Stairs)', fontsize=14, fontweight='bold', color='#c0392b')
    axes[2].set_xlabel('True Heart Rate (ECG)')
    axes[2].set_xlim(axis_min, axis_max)
    axes[2].set_ylim(axis_min, axis_max)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/kmeans_ecg_3way_compare.png'
    plt.savefig(save_path, dpi=300)
    print(f"[v] 三方散佈圖已儲存: {save_path}")

if __name__ == "__main__":
    run_kmeans_ecg_validation()