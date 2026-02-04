import os
import data_loader
import feature_extractor
import clustering_model

# ================= 設定區 =================
FILE_PATH = "F:/碩士/114-1_資料探勘/archive (1)/PPG_FieldStudy/S7/S7.pkl"
TEST_LIMIT = None
OUTPUT_DIR = "output"
CSV_NAME = 'clustering_results.csv'
# =========================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Step 1: 讀取資料
    print("--- Step 1: Data Loading ---")
    ppg_signal, labels, _, fs = data_loader.load_ppg_data(FILE_PATH)
    
    if ppg_signal is not None:
        # 畫波形圖
        data_loader.plot_raw_wave(
            ppg_signal, fs, duration=15, 
            save_path=os.path.join(OUTPUT_DIR, '01_raw_waveform.png')
        )
        
        # Step 2: 特徵提取
        print("\n--- Step 2: Feature Extraction ---")
        df_features = feature_extractor.extract_features(
            ppg_signal, labels, fs, limit=TEST_LIMIT
        )
        
        # Step 3: 分群
        print("\n--- Step 3: Clustering ---")
        # 執行分群並畫圖
        clustering_model.run_clustering_and_plot(
            df_features, 
            save_path=os.path.join(OUTPUT_DIR, '02_clustering.png')
        )
        
        # 將結果存成 CSV 檔
        csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
        df_features.to_csv(csv_path, index=False)
        print(f"\n[v] 分群結果已儲存至: {csv_path}")
        print("現在可以執行 validation 來查看驗證結果了！")

if __name__ == "__main__":
    main()