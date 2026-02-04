import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_ppg_data(file_path):
    print(f"正在讀取檔案: {file_path} ...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # 1. 讀取 PPG
        ppg_signal = data['signal']['wrist']['BVP'].flatten()
        
        # 2. 讀取 Activity
        if 'activity' in data:
            activity_signal = data['activity'].flatten()
        else:
            print("警告：找不到 'activity' 欄位！")
            return None, None, None, None
        # === 3. 讀取 ECG 心率真值 (Label) ===
        # 根據 Readme，這裡存的是 8秒視窗/2秒位移 的 ECG 心率 
        if 'label' in data:
            ecg_hr_label = data['label'].flatten()
        else:
            print("警告：找不到 'label' (ECG HR) 欄位！")
            # 如果找不到 ECG Label, 傳回空的陣列
            ecg_hr_label = np.array([]) 

        fs_ppg = 64
        print(f"讀取成功！PPG: {len(ppg_signal)}, Activity: {len(activity_signal)}, ECG Label: {len(ecg_hr_label)}")
        
        # 回傳 4 個變數: PPG, Activity, ECG Label, fs
        return ppg_signal, activity_signal, ecg_hr_label, fs_ppg

    except Exception as e:
        print(f"讀取錯誤: {e}")
        return None, None, None, None

    except Exception as e:
        print(f"讀取錯誤: {e}")
        return None, None, None
    except Exception as e:
        print(f"讀取錯誤: {e}")
        # 如果失敗，印出所有的 keys 幫助除錯
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            print(f"檔案結構 keys: {data.keys()}")
        except:
            pass
        return None, None, None

def plot_raw_wave(signal, fs, duration=15, save_path='raw_wave.png'):
    """
    繪製並儲存原始波形
    """
    samples = duration * fs
    time_axis = np.arange(samples) / fs

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, signal[:samples], color='#2c3e50', linewidth=1.2)
    plt.title(f'Raw PPG Signal (First {duration}s)', fontsize=14)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    

    plt.savefig(save_path, dpi=300)
    print(f">> 波形圖已儲存至: {save_path}")
    plt.close() # 關閉圖表