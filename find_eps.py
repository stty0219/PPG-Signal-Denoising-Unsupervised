import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

# ================= 設定區 =================
CSV_PATH = 'output/clustering_results.csv'
MinPts = 20
OUTPUT_FILE = 'output/auto_k_distance_plot.png'
# =========================================

def find_optimal_epsilon_and_plot():
    print(f"=== 自動尋找最佳 Epsilon (Knee Point Detection) ===")
    
    # 1. 讀取資料
    if not os.path.exists(CSV_PATH):
        print(f"錯誤: 找不到 {CSV_PATH}")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        # 確保使用完整的 5 維特徵
        feature_cols = ['SpecEn', 'PermEn', 'SampEn', 'Petrosian_FD', 'Higuchi_FD']
        
        # 簡單檢查欄位
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) < 5:
            print(f"警告: 缺少部分特徵，目前使用: {available_cols}")
        
        df_clean = df.dropna(subset=available_cols)
        
        # 標準化 (非常重要，否則距離計算會失準)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[available_cols])
        
    except Exception as e:
        print(f"資料處理錯誤: {e}")
        return

    # 2. 計算 K-Nearest Neighbors 距離
    print(">> 計算 K-distance 曲線...")
    nbrs = NearestNeighbors(n_neighbors=MinPts).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    
    # 取出第 k 個鄰居的距離並排序
    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)

    # 3. 自動尋找轉折點 (幾何距離法)
    # 為了避免 X軸(點的數量) 和 Y軸(距離數值) 的尺度差異影響計算，先將 X 和 Y 都歸一化 (Normalize) 到 [0, 1] 之間
    
    n_points = len(k_distances)
    indices_norm = np.arange(n_points) / (n_points - 1)
    dists_norm = (k_distances - k_distances.min()) / (k_distances.max() - k_distances.min())
    
    # 定義起點與終點的連線向量 (Line Vector)
    # 起點 (0, 0), 終點 (1, 1) -> 因為已經歸一化且排序過
    vec_line = np.array([1, 1])
    
    # 計算每個點到這條對角線的垂直距離
    # 點 P 的向量為 (x_norm, y_norm)
    # 距離 d = |x_norm - y_norm| / sqrt(2) (利用向量外積原理)
    vec_points = np.vstack((indices_norm, dists_norm)).T
    
    # 計算距離：在 2D 平面，點到線的距離正比於 |y - x| (因為線是 y=x)
    # 這裡我們找 "距離對角線最遠" 的點
    dist_to_line = np.abs(dists_norm - indices_norm) / np.sqrt(2)
    
    # 找出最大距離的索引 (Knee Point)
    knee_idx = np.argmax(dist_to_line)
    optimal_eps = k_distances[knee_idx]
    
    print(f">> [成功] 偵測到最佳轉折點: Index={knee_idx}, Epsilon={optimal_eps:.4f}")

    # 4. 繪圖
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, linewidth=2, label='K-distance curve')
    
    # 標記轉折點
    plt.plot(knee_idx, optimal_eps, 'ro', markersize=8, label='Elbow Point')
    
    # 畫出建議的 Epsilon 線
    plt.axhline(y=optimal_eps, color='r', linestyle='--', linewidth=1.5, label=f'Optimal eps={optimal_eps:.3f}')
    
    # 加入文字註解
    plt.annotate(f'Best eps={optimal_eps:.3f}', 
                 xy=(knee_idx, optimal_eps), 
                 xytext=(knee_idx - n_points*0.25, optimal_eps + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, fontweight='bold')

    plt.title(f'K-distance Graph with Auto-Knee Detection (MinPts={MinPts})', fontsize=14)
    plt.xlabel('Points sorted by distance', fontsize=12)
    plt.ylabel(f'{MinPts}-th Nearest Neighbor Distance (Epsilon)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"[v] 圖表已儲存至: {OUTPUT_FILE}")
    print(f"*** DBSCAN 建議參數為: eps={optimal_eps:.2f} ***")

if __name__ == "__main__":
    find_optimal_epsilon_and_plot()