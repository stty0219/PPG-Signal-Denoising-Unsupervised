import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import data_loader 

# ================= 設定區 =================
CSV_PATH = 'output/clustering_results.csv'
RAW_FILE_PATH = "F:/碩士/114-1_資料探勘/archive (1)/PPG_FieldStudy/S7/S7.pkl"
OUTPUT_DIR = "output"
# =========================================

def plot_psd_comparison():
    print("=== 正在進行頻譜分析 (基於 Fourier Transform) ===")
    
    # 1. 讀取資料
    try:
        df = pd.read_csv(CSV_PATH)
        raw_signal, _, _, fs = data_loader.load_ppg_data(RAW_FILE_PATH)
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return

    print(">> 重建 K-Means 模型...")
    
    # 確保特徵沒有空值
    feature_cols = ['SpecEn', 'PermEn', 'Petrosian_FD', 'Higuchi_FD']
    df = df.dropna(subset=feature_cols)
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 執行 K-Means
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    df['Cluster_K8'] = kmeans.fit_predict(X_scaled)
    # ==========================================

    # 2. 設定要比較的群集 ID
    clean_id = 2
    noisy_id = 1
    
    # 找出樣本索引
    clean_indices = df[df['Cluster_K8'] == clean_id].index
    noisy_indices = df[df['Cluster_K8'] == noisy_id].index
    
    if len(clean_indices) == 0 or len(noisy_indices) == 0:
        print(f"錯誤: 找不到 Cluster {clean_id} 或 {noisy_id} 的資料。")
        print(f"目前的 Cluster ID: {df['Cluster_K8'].unique()}")
        return

    # 3. 隨機抽取樣本
    np.random.seed(42)
    idx_clean = np.random.choice(clean_indices)
    idx_noisy = np.random.choice(noisy_indices)
    
    # 取得 8 秒波形
    start_c = int(df.loc[idx_clean, 'Start_Index'])
    start_n = int(df.loc[idx_noisy, 'Start_Index'])
    window = 8 * 64
    
    wave_clean = raw_signal[start_c : start_c + window]
    wave_noisy = raw_signal[start_n : start_n + window]
    
    # 4. 核心步驟：計算 PSD (Welch 方法)
    print(">> 計算功率譜密度 (PSD)...")
    f_c, Pxx_c = welch(wave_clean, fs, nperseg=512)
    f_n, Pxx_n = welch(wave_noisy, fs, nperseg=512)
    
    # 5. 繪圖對決
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- 上：Cluster Clean (Regular Heartbeat) ---
    # 時域
    axes[0,0].plot(wave_clean, color='#2ecc71', linewidth=1.5)
    axes[0,0].set_title(f'Cluster {clean_id} - Time Domain\n(Regular Heartbeat)', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].grid(True, alpha=0.3)
    
    # 頻域
    axes[0,1].plot(f_c, Pxx_c, color='#2ecc71', linewidth=2)
    axes[0,1].set_title(f'Cluster {clean_id} - PSD Spectrum', fontsize=12, fontweight='bold')
    axes[0,1].set_xlim(0, 5) # 只看 0-5Hz
    axes[0,1].set_ylabel('Power')
    axes[0,1].grid(True, alpha=0.3)
    # 標示主頻
    peak_freq_c = f_c[np.argmax(Pxx_c)]
    axes[0,1].axvline(x=peak_freq_c, color='black', linestyle='--', alpha=0.5)
    axes[0,1].text(peak_freq_c+0.1, np.max(Pxx_c)*0.9, f'HR: {peak_freq_c*60:.0f} BPM', color='black')

    # --- 下：Cluster Noisy (Irregular/Spiky) ---
    # 時域
    axes[1,0].plot(wave_noisy, color='#e74c3c', linewidth=1.5)
    axes[1,0].set_title(f'Cluster {noisy_id} - Time Domain\n(Irregular/Spiky)', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Time samples')
    axes[1,0].set_ylabel('Amplitude')
    axes[1,0].grid(True, alpha=0.3)
    
    # 頻域
    axes[1,1].plot(f_n, Pxx_n, color='#e74c3c', linewidth=2)
    axes[1,1].set_title(f'Cluster {noisy_id} - PSD Spectrum', fontsize=12, fontweight='bold')
    axes[1,1].set_xlim(0, 5)
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].grid(True, alpha=0.3)
    # 標示主頻
    peak_freq_n = f_n[np.argmax(Pxx_n)]
    axes[1,1].axvline(x=peak_freq_n, color='black', linestyle='--', alpha=0.5)
    axes[1,1].text(peak_freq_n+0.1, np.max(Pxx_n)*0.9, f'HR: {peak_freq_n*60:.0f} BPM', color='black')

    plt.tight_layout()
    save_file = f'{OUTPUT_DIR}/K-means_spectral_analysis.png'
    plt.savefig(save_file, dpi=300)
    print(f"K-Means 頻譜分析圖已儲存: {save_file}")
    
if __name__ == "__main__":
    plot_psd_comparison()