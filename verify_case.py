import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# 1. 載入本機已有的正式模型與 Scaler
model = tf.keras.models.load_model("lstm_model_multivariate.h5", compile=False)
scaler = joblib.load("data_scaler_multivariate.pkl")

def run_single_day_verification(history_csv_path, pv_csv_path, target_date="2026-10-19"):
    """
    執行單日完整驗證案例：明確切分歷史輸入與未來真值，並計算真實誤差
    """
    print(f"🚀 開始執行 【{target_date}】 之正式微電網推論與驗證流程...")
    
    # 2. 讀取真實資料檔案
    df_history = pd.read_csv(history_csv_path)  # 包含預測前 96 筆歷史資料
    df_pv = pd.read_csv(pv_csv_path)            # 包含當天 96 筆光伏預報
    
    # 3. 欄位與格式檢查
    required_cols = ['timestamp', 'load_kw', 'temperature', 'ghi']
    for c in required_cols:
        if c not in df_history.columns:
            raise ValueError(f"歷史資料缺少必要欄位: {c}")
            
    # 自動適應正式 PV 欄位名稱 (支援 P_PV_kW 或 P_PV_kW_DEMO)[cite: 5]
    pv_col = 'P_PV_kW' if 'P_PV_kW' in df_pv.columns else 'P_PV_kW_DEMO'
    time_col = 'IntervalEnding' if 'IntervalEnding' in df_pv.columns else 'timestamp'
    
    # 4. 取出預測前最後 96 筆歷史矩陣作為 Model Input [load_kw, temperature, ghi][cite: 5, 7]
    arr_hist = df_history[['load_kw', 'temperature', 'ghi']].tail(96).to_numpy(dtype=float)
    if len(arr_hist) < 96:
        raise ValueError(f"歷史資料長度不足 96 筆，目前僅有 {len(arr_hist)} 筆")
        
    # 標準化與 3D 模型推論 (Lookback = 96)[cite: 5, 7]
    scaled_hist = scaler.transform(arr_hist)
    input_tensor = scaled_hist.reshape(1, 96, 3)
    scaled_pred = model.predict(input_tensor, verbose=0)
    
    # 反還原預測總負載 (Gross Load)[cite: 5, 7]
    dummy = np.zeros((96, 3))
    dummy[:, 0] = scaled_pred[0]
    gross_load_pred = scaler.inverse_transform(dummy)[:, 0]
    
    # 5. 取得對應時間戳與 PV 數值
    timestamps = pd.to_datetime(df_pv[time_col]).tail(96).reset_index(drop=True)
    pv_vals = pd.to_numeric(df_pv[pv_col], errors='coerce').tail(96).reset_index(drop=True).to_numpy(float)
    
    # 6. 計算淨負載：Net Load (kW) = Gross Load - PV Forecast[cite: 5]
    net_load_pred = gross_load_pred - pv_vals
    
    # 7. 建立標準輸出 DataFrame
    out_df = pd.DataFrame({
        "timestamp": timestamps,
        "gross_load_kw": np.round(gross_load_pred, 2),
        "pv_kw": np.round(pv_vals, 2),
        "net_load_kw": np.round(net_load_pred, 2)
    })
    
    # 8. 如果歷史/真值檔案中包含當天實測值 (Ground Truth)，進行真實精度核對
    if 'actual_load_kw' in df_history.columns:
        actual_vals = df_history['actual_load_kw'].tail(96).reset_index(drop=True).to_numpy(float)
        out_df["actual_load_kw"] = actual_vals
        
        # 計算真實誤差指標
        mape = mean_absolute_percentage_error(actual_vals, gross_load_pred) * 100
        accuracy = 100 - mape
        mae = mean_absolute_error(actual_vals, gross_load_pred)
        rmse = np.sqrt(mean_squared_error(actual_vals, gross_load_pred))
        
        print("\n" + "="*50)
        print(f"📊 【{target_date}】 真實預測精度核對報告")
        print("="*50)
        print(f"• 平均絕對百分比誤差 (MAPE): {mape:.2f}%")
        print(f"• 整體預測準確度 (Accuracy):   {accuracy:.2f}%")
        print(f"• 平均絕對誤差 (MAE):        {mae:.2f} kW")
        print(f"• 均方根誤差 (RMSE):         {rmse:.2f} kW")
        print("="*50)
    else:
        print("⚠️ 註記：目前提供的資料檔尚未包含當天實測值 (actual_load_kw)，僅輸出推論與淨負載結果。")
        
    # 9. 匯出正式輸出檔
    output_filename = f"Verified_EMS_Output_{target_date}.csv"
    out_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"📁 已成功匯出可核對來源的正式輸出檔：{output_filename}")
    
    return out_df

# 【執行範例】（請將檔名替換為組員提供的真實檔案）
# verified_result = run_single_day_verification("history_20261019.csv", "PV_Forecast_15min_20261019.csv", "2026-10-19")