import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import data_loader 

# ================= 設定區 =================
CSV_PATH = 'output/clustering_results.csv'
RAW_FILE_PATH = "F:/碩士/114-1_資料探勘/archive (1)/PPG_FieldStudy/S7/S7.pkl"
OUTPUT_DIR = "output"
# =========================================

def plot_dbscan_spectral_comparison():
    print("=== 正在執行 DBSCAN 頻譜分析 (Spectral Analysis) ===")
    
    # 1. 讀取資料
    try:
        df = pd.read_csv(CSV_PATH)
        raw_signal, _, _, fs = data_loader.load_ppg_data(RAW_FILE_PATH)
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return

    print(">> 重建 DBSCAN 模型...")
    
    feature_cols = ['SpecEn', 'PermEn', 'Petrosian_FD', 'Higuchi_FD']
    df = df.dropna(subset=feature_cols)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 使用你設定的參數
    dbscan = DBSCAN(eps=0.3, min_samples=20)
    df['DBSCAN_Label'] = dbscan.fit_predict(X_scaled)

    # 統計一下分群結果
    print(f"分群統計:\n{df['DBSCAN_Label'].value_counts()}")

    # 2. 設定要比較的群集 ID
    # DBSCAN 邏輯: 0 是最大群 (Core), -1 是雜訊 (Noise)
    core_id = 0 
    noise_id = 1
    
    core_indices = df[df['DBSCAN_Label'] == core_id].index
    noise_indices = df[df['DBSCAN_Label'] == noise_id].index
    
    if len(core_indices) == 0 or len(noise_indices) == 0:
        print(f"錯誤: 找不到 Label {core_id} 或 {noise_id} 的資料。")
        return

    # 3. 隨機抽取樣本
    np.random.seed(42) 
    idx_core = np.random.choice(core_indices)
    idx_noise = np.random.choice(noise_indices)
    
    # 取得 8 秒波形
    start_c = int(df.loc[idx_core, 'Start_Index'])
    start_n = int(df.loc[idx_noise, 'Start_Index'])
    window = 8 * 64
    
    wave_core = raw_signal[start_c : start_c + window]
    wave_noise = raw_signal[start_n : start_n + window]
    
    # 4. 計算 PSD (Welch 方法)
    print(">> 計算功率譜密度 (PSD)...")
    f_c, Pxx_c = welch(wave_core, fs, nperseg=512)
    f_n, Pxx_n = welch(wave_noise, fs, nperseg=512)
    
    # 5. 繪圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- 上：DBSCAN Core Cluster (0) ---
    # 時域
    axes[0,0].plot(wave_core, color='#3498db', linewidth=1.5)
    axes[0,0].set_title(f'DBSCAN Label {core_id} (Core) - Time Domain', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].grid(True, alpha=0.3)
    
    # 頻域
    axes[0,1].plot(f_c, Pxx_c, color='#3498db', linewidth=2)
    axes[0,1].set_title(f'DBSCAN Label {core_id} (Core) - PSD Spectrum', fontsize=12, fontweight='bold')
    axes[0,1].set_xlim(0, 5) 
    axes[0,1].set_ylabel('Power')
    axes[0,1].grid(True, alpha=0.3)
    # 標示主頻
    peak_freq_c = f_c[np.argmax(Pxx_c)]
    axes[0,1].axvline(x=peak_freq_c, color='black', linestyle='--', alpha=0.5)
    axes[0,1].text(peak_freq_c+0.1, np.max(Pxx_c)*0.9, f'HR: {peak_freq_c*60:.0f} BPM', color='black')

    # --- 下：DBSCAN Noise (-1) ---
    # 時域
    axes[1,0].plot(wave_noise, color='#e74c3c', linewidth=1.5)
    axes[1,0].set_title(f'DBSCAN Label {noise_id} (Noise) - Time Domain', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Time samples')
    axes[1,0].set_ylabel('Amplitude')
    axes[1,0].grid(True, alpha=0.3)
    
    # 頻域
    axes[1,1].plot(f_n, Pxx_n, color='#e74c3c', linewidth=2)
    axes[1,1].set_title(f'DBSCAN Label {noise_id} (Noise) - PSD Spectrum', fontsize=12, fontweight='bold')
    axes[1,1].set_xlim(0, 5)
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].grid(True, alpha=0.3)
    # 標示主頻
    peak_freq_n = f_n[np.argmax(Pxx_n)]
    axes[1,1].axvline(x=peak_freq_n, color='black', linestyle='--', alpha=0.5)
    axes[1,1].text(peak_freq_n+0.1, np.max(Pxx_n)*0.9, f'HR: {peak_freq_n*60:.0f} BPM', color='black')

    plt.tight_layout()
    save_file = f'{OUTPUT_DIR}/dbscan_spectral_analysis.png'
    plt.savefig(save_file, dpi=300)
    print(f"DBSCAN 頻譜分析圖已儲存: {save_file}")

if __name__ == "__main__":
    plot_dbscan_spectral_comparison()