import numpy as np
import pandas as pd
import antropy as ant
from scipy import stats # 用來算眾數 (mode)

def extract_features(signal, labels, fs, window_sec=8, shift_sec=2, limit=None):
    """
    新增參數: labels (活動標籤陣列)
    """
    window_size = window_sec * fs
    step_size = shift_sec * fs
    
    # 計算 PPG (64Hz) 與 Label (4Hz) 的採樣率比例
    # PPG-DaLiA: 64 / 4 = 16 (每 16 個 PPG 點對應 1 個 Label 點)
    ratio = 16 

    features_list = []
    
    print(f"開始特徵提取...")

    for i in range(0, len(signal) - window_size, step_size):
        # 1. 切割 PPG 訊號
        segment = signal[i : i + window_size]
        
        # 2. 處理對應的 Label (找出這 8 秒內做最多的活動)
        label_start = i // ratio
        label_end = (i + window_size) // ratio
        segment_labels = labels[label_start : label_end]
        
        if len(segment_labels) == 0: continue
        
        # 取眾數 (Mode) 作為這段視窗的 Ground Truth
        mode_res = stats.mode(segment_labels, keepdims=True)
        activity_id = mode_res.mode[0] 

        # 標準化
        if np.std(segment) == 0: continue
        segment_norm = (segment - np.mean(segment)) / np.std(segment)
        
        try:
            # === 特徵計算 (維持原樣) ===
            spec_en = ant.spectral_entropy(segment_norm, fs, method='welch', normalize=True)
            perm_en = ant.perm_entropy(segment_norm, normalize=True)
            samp_en = ant.sample_entropy(segment_norm)
            
            petro_fd = ant.petrosian_fd(segment_norm)
            higuchi_fd = ant.higuchi_fd(segment_norm)
            
            features_list.append({
                'Start_Index': i,       # 記錄原始訊號的起始點
                'Activity_Label': activity_id, # 記錄 Ground Truth
                'SpecEn': spec_en,
                'PermEn': perm_en,
                'SampEn': samp_en,
                'Petrosian_FD': petro_fd,
                'Higuchi_FD': higuchi_fd,
            })
            
        except Exception as e:
            continue

        if limit and len(features_list) >= limit:
            break
            
    return pd.DataFrame(features_list)