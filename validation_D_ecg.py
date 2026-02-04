import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import data_loader 

# ================= 設定區 =================
CSV_PATH = 'output/clustering_results.csv'
RAW_FILE_PATH = "F:/碩士/114-1_資料探勘/archive (1)/PPG_FieldStudy/S7/S7.pkl"
OUTPUT_DIR = "output"
# =========================================

def calculate_ppg_hr(signal_segment, fs=64):
    """ 計算 PPG 主頻 (BPM) """
    f, Pxx = welch(signal_segment, fs, nperseg=512)
    # 限制在 30-240 BPM 範圍
    valid_mask = (f >= 0.5) & (f <= 4.0)
    f_valid = f[valid_mask]
    Pxx_valid = Pxx[valid_mask]
    
    if len(Pxx_valid) == 0: return np.nan
    peak_freq = f_valid[np.argmax(Pxx_valid)]
    return peak_freq * 60

def run_dbscan_ecg_validation():
    print("=== 正在執行 DBSCAN 心率誤差驗證 (ECG Validation) ===")
    
    # 1. 讀取資料
    try:
        df = pd.read_csv(CSV_PATH)
        ppg_signal, _, ecg_labels, fs = data_loader.load_ppg_data(RAW_FILE_PATH)
        
        if ecg_labels is None:
            print("錯誤：無法讀取 ECG Label")
            return
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return

    # 2. 重建 DBSCAN
    print(">> 重建 DBSCAN 模型...")
    feature_cols = ['SpecEn', 'PermEn', 'Petrosian_FD', 'Higuchi_FD']
    df = df.dropna(subset=feature_cols)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # 使用你設定的參數
    dbscan = DBSCAN(eps=0.3, min_samples=20)
    df['DBSCAN_Label'] = dbscan.fit_predict(X_scaled)

    # 3. 計算心率誤差
    print(f">> 開始計算 {len(df)} 筆資料的心率誤差...")
    
    ppg_bpms = []
    ecg_bpms = []
    errors = []
    
    for idx, row in df.iterrows():
        start_idx = int(row['Start_Index'])
        
        # 取得 PPG 片段
        segment = ppg_signal[start_idx : start_idx + 8*fs]
        
        # 計算 PPG HR
        ppg_hr = calculate_ppg_hr(segment, fs)
        
        # 取得 ECG Truth (Label 2秒一筆)
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
    
    # 去除空值
    df_final = df.dropna(subset=['HR_Error'])
    
    # 4. 統計 MAE
    print("\n=== DBSCAN 各群集心率誤差統計 (MAE) ===")
    mae_stats = df_final.groupby('DBSCAN_Label')['HR_Error'].agg(['mean', 'std', 'count', 'min', 'max'])
    print(mae_stats.round(2))

    # 5. 繪圖
    
    # --- 圖表 A: 箱型圖 ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='DBSCAN_Label', y='HR_Error', data=df_final, palette="Reds", showfliers=False)
    plt.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='Acceptable Error (10 BPM)')
    plt.title('DBSCAN Heart Rate Error Distribution (-1 is Noise)', fontsize=14)
    plt.xlabel('DBSCAN Label')
    plt.ylabel('Absolute Error (BPM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dbscan_ecg_boxplot.png', dpi=300)
    print(f"[v] 誤差箱型圖已儲存: {OUTPUT_DIR}/dbscan_ecg_boxplot.png")
    
    # --- 圖表 B: 散佈圖 ---
    print(">> 正在繪製三方散佈圖...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 設定統一的座標軸範圍
    axis_min, axis_max = 40, 180
    
    # --- 圖 1: 核心群 Label 0 (Baseline) ---
    data_0 = df_final[df_final['DBSCAN_Label'] == 0]
    axes[0].scatter(data_0['ECG_HR'], data_0['PPG_HR'], alpha=0.5, color='#3498db', label='Core Points')
    axes[0].plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7)
    axes[0].set_title(f'Label 0: Core Cluster\n(Predicted: Normal)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('True Heart Rate (ECG)')
    axes[0].set_ylabel('Estimated Heart Rate (PPG)')
    axes[0].set_xlim(axis_min, axis_max)
    axes[0].set_ylim(axis_min, axis_max)
    axes[0].grid(True, alpha=0.3)
    
    # --- 圖 2: 雜訊群 Label -1 (False Rejection) ---
    data_noise = df_final[df_final['DBSCAN_Label'] == -1]
    axes[1].scatter(data_noise['ECG_HR'], data_noise['PPG_HR'], alpha=0.5, color='#e74c3c', label='Noise Points')
    axes[1].plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7)
    axes[1].set_title(f'Label -1: Noise\n(Predicted: Noise -> But Good!)', fontsize=14, fontweight='bold', color='#c0392b')
    axes[1].set_xlabel('True Heart Rate (ECG)')
    axes[1].set_xlim(axis_min, axis_max)
    axes[1].set_ylim(axis_min, axis_max)
    axes[1].grid(True, alpha=0.3)
    
    # --- 圖 3: 錯誤群 Label 1 (False Acceptance) ---
    data_1 = df_final[df_final['DBSCAN_Label'] == 1]
    axes[2].scatter(data_1['ECG_HR'], data_1['PPG_HR'], alpha=0.5, color='#e67e22', label='Cluster 1')
    axes[2].plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7)
    axes[2].set_title(f'Label 1: High Density Cluster\n(Predicted: Normal -> But Bad!)', fontsize=14, fontweight='bold', color='#d35400')
    axes[2].set_xlabel('True Heart Rate (ECG)')
    axes[2].set_xlim(axis_min, axis_max)
    axes[2].set_ylim(axis_min, axis_max)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/dbscan_ecg_3way_compare.png'
    plt.savefig(save_path, dpi=300)
    print(f"[v] 三方散佈圖已儲存: {save_path}")

if __name__ == "__main__":
    run_dbscan_ecg_validation()