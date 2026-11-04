import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

print("🚀 開始執行嚴格對齊真實 PV 檔的 EMS 輸出與淨負載重算...")

# 1. 載入模型與 Scaler (3 特徵版本: load_kw, temperature, ghi)
model = tf.keras.models.load_model("lstm_model_multivariate.h5", compile=False)
scaler = joblib.load("data_scaler_multivariate.pkl")

# 2. 直接讀取你們提供的真實光伏預報檔
pv_filename = "PV_Forecast_15min_20261019.csv"
df_pv_raw = pd.read_csv(pv_filename)

# 確保欄位對齊為 P_PV_kW (原先為 P_PV_kW_DEMO)
if 'P_PV_kW_DEMO' in df_pv_raw.columns:
    df_pv_raw = df_pv_raw.rename(columns={'P_PV_kW_DEMO': 'P_PV_kW'})

# 提取 96 筆 PV 數值與時間戳 (IntervalEnding 代表區間結束)
timestamps = pd.to_datetime(df_pv_raw['IntervalEnding']).tail(96).reset_index(drop=True)
pv_vals = pd.to_numeric(df_pv_raw['P_PV_kW'], errors='coerce').tail(96).reset_index(drop=True).to_numpy(float)

# 3. 讀取實場歷史用電作為特徵來源 (例如從 Report 抓取人言大樓歷史 kw)
df_building = pd.read_csv("Report251014_m_buildings.csv")
hist_load = df_building['人言大樓kw'].tail(96).to_numpy(dtype=float)

# 4. 建立 3 特徵矩陣 [load_kw, temperature, ghi]
# 氣溫與 GHI 依時間軸進行合理物理模擬補齊
hist_temp = 25.0 + 3.0 * np.sin(np.linspace(0, 2 * np.pi, 96))
hist_ghi = np.maximum(0, 800 * np.sin(np.linspace(0, np.pi, 96)))
arr_hist = np.column_stack([hist_load, hist_temp, hist_ghi])

# 5. 執行 3 特徵 LSTM 模型推論，還原總負載 (Gross Load)
scaled_hist = scaler.transform(arr_hist)
input_tensor = scaled_hist.reshape(1, 96, 3)
scaled_pred = model.predict(input_tensor, verbose=0)

dummy = np.zeros((96, 3))
dummy[:, 0] = scaled_pred[0]
gross_load_pred = scaler.inverse_transform(dummy)[:, 0]

# 6. 嚴格依同一份 PV 檔重算淨負載：Net Load = Gross Load - PV
net_load_pred = gross_load_pred - pv_vals

# 7. 建立符合標準的最終輸出 DataFrame
out_df = pd.DataFrame({
    "timestamp": timestamps,          # 代表區間結束時間 (IntervalEnding)
    "gross_load_kw": np.round(gross_load_pred, 2),
    "pv_kw": np.round(pv_vals, 2),    # 直接對應你們提供的真實 PV 檔數值
    "net_load_kw": np.round(net_load_pred, 2)
})

# 8. 匯出正式對接 CSV
output_filename = "EMS_Output_Synced_PV_20261019.csv"
out_df.to_csv(output_filename, index=False, encoding="utf-8-sig")

print(f"🎉 成功產出與你們 PV 檔同步的正式輸出檔：{output_filename}")
print(out_df.head())