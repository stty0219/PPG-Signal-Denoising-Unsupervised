import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

def run_clustering_and_plot(df, save_path='clustering_result.png'):
    """
    執行分群並儲存結果圖
    """
    if df is None or df.empty:
        print("錯誤：沒有特徵資料可以分群")
        return

    print("正在進行資料標準化與分群...")
    
    # 1. 資料標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # 2. K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['KMeans_Labels'] = kmeans.fit_predict(X_scaled)
    
    # 3. DBSCAN
    dbscan = DBSCAN(eps=0.75, min_samples=20)
    df['DBSCAN_Labels'] = dbscan.fit_predict(X_scaled)
    
    # 4. 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # K-Means Plot
    sns.scatterplot(data=df, x='SampEn', y='Higuchi_FD', 
                    hue='KMeans_Labels', palette='viridis', ax=axes[0], alpha=0.7)
    axes[0].set_title('Method 1: K-Means (K= ' + f'{kmeans.n_clusters})', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # DBSCAN Plot
    unique_labels = df['DBSCAN_Labels'].unique()
    # 自動調整顏色數量
    palette_name = 'deep' if len(unique_labels) <= 10 else 'tab20'
    
    sns.scatterplot(data=df, x='SampEn', y='Higuchi_FD', 
                    hue='DBSCAN_Labels', palette=palette_name, ax=axes[1], alpha=0.7)
    axes[1].set_title('Method 2: DBSCAN (Outlier Detection)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    

    plt.savefig(save_path, dpi=300)
    print(f">> 分群結果圖已儲存至: {save_path}")
    plt.close() # 關閉圖表
    
    # 5. 文字報告
    print("\n=== 分群結果統計 ===")
    print(f"K-Means 各群數量:\n{df['KMeans_Labels'].value_counts()}")
    print("\n------------------")
    print(f"DBSCAN 各群數量 (-1 為雜訊):\n{df['DBSCAN_Labels'].value_counts()}")