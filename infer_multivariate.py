# =========================================================
# 微電網 EMS 系統 - 多變數負載預測推論與防禦模組 (infer_multivariate.py - 修正版)
# 輸入特徵嚴格順序：[load_kw, temperature, ghi]
# 模型：回看 L 步（預設 96）-> 一次輸出 B 步總負載（預設 96）
# 修正重點：
#   1. 一次預測直接取全部輸出（不再逐步預測、錯位取值）
#   2. 超過 1 天時，以「整個區塊」為單位滾動
#   3. NaN / 負值 / 超出範圍的輸出一律走備援，不再悄悄改成 0
#   4. 備援會真的算出值；加入氣溫、GHI 範圍檢查與時間間隔檢查
#   5. 新增 compute_net_load()：對齊時間、扣除 PV，淨負載不截斷負值
# =========================================================

import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

STEPS_PER_DAY = 96           # 24 小時 x 4 筆/小時
INTERVAL_MIN = 15
MAX_FORECAST_DAYS = 30
N_FEATURES = 3
FEATURE_ORDER = ["load_kw", "temperature", "ghi"]
TEMP_RANGE = (-20.0, 50.0)   # 氣溫合理範圍 (°C)
GHI_RANGE = (0.0, 1500.0)    # 日照合理範圍 (W/m²)
LOAD_UPPER_FACTOR = 3.0      # 預測負載超過訓練最大值的 3 倍視為異常


class MultivariateEMSpredictor:
    def __init__(self,
                 model_path='lstm_model_multivariate.h5',
                 scaler_path='data_scaler_multivariate.pkl'):
        base_dir = Path(__file__).resolve().parent
        self.model_path = str(base_dir / model_path)
        self.scaler_path = str(base_dir / scaler_path)

        self.model = None
        self.scaler = None
        self.is_ready = False
        self.model_version = "multivariate_v1_weather"
        self.target_unit = "kW"
        self.init_error = None
        self.lookback_steps = STEPS_PER_DAY   # 載入模型後以模型實際形狀覆蓋
        self.block_steps = STEPS_PER_DAY      # 模型一次輸出的步數
        self.load_upper = None

        self.stats = {
            "total_requests": 0,
            "errors": {
                "timestamp_error": 0,
                "horizon_error": 0,
                "dimension_error": 0,
                "length_insufficient": 0,
                "invalid_input": 0,
                "negative_load_blocked": 0,
                "weather_error": 0,
                "interval_error": 0,
                "model_output_nan": 0,
                "model_output_incomplete": 0,
                "negative_output_blocked": 0,
                "model_output_out_of_range": 0,
                "model_not_ready": 0,
                "unexpected_exception": 0
            },
            "fallback_used": {
                "historical_matching": 0,
                "moving_average": 0,
                "none": 0
            }
        }
        self._load_components()

    # ------------------------------------------------------
    def _load_components(self):
        try:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"找不到多變數 Scaler: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            n_in = getattr(self.scaler, "n_features_in_", N_FEATURES)
            if n_in != N_FEATURES:
                raise ValueError(f"Scaler 欄位數為 {n_in}，應為 {N_FEATURES} ({FEATURE_ORDER})")
            data_max = getattr(self.scaler, "data_max_", None)
            if data_max is not None:
                self.load_upper = float(data_max[0]) * LOAD_UPPER_FACTOR

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"找不到多變數模型檔: {self.model_path}")
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path, compile=False)

            in_shape, out_shape = self.model.input_shape, self.model.output_shape
            if len(in_shape) != 3 or in_shape[2] != N_FEATURES:
                raise ValueError(f"模型輸入形狀應為 (None, L, {N_FEATURES})，實際為 {in_shape}")
            if len(out_shape) != 2:
                raise ValueError(f"模型輸出形狀應為 (None, B)，實際為 {out_shape}")
            if in_shape[1]:
                self.lookback_steps = int(in_shape[1])
            if out_shape[1]:
                self.block_steps = int(out_shape[1])
            self.is_ready = True
        except Exception as e:
            self.is_ready = False
            self.init_error = f"多變數元件載入失敗: {str(e)}"

    def _inverse_load(self, scaled_vec):
        """只還原負載欄（第 0 欄）；其餘欄位補 0 不影響結果（MinMax 為逐欄運算）。"""
        dummy = np.zeros((len(scaled_vec), N_FEATURES))
        dummy[:, 0] = scaled_vec
        return self.scaler.inverse_transform(dummy)[:, 0]

    @staticmethod
    def _make_timestamps(parsed_dt, n):
        return [(parsed_dt + timedelta(minutes=INTERVAL_MIN) * (i + 1)).isoformat()
                for i in range(n)]

    # ------------------------------------------------------
    def forecast_load_multivariate(self, recent_history_matrix, latest_data_timestamp,
                                   forecast_days=1, future_weather_matrix=None,
                                   history_timestamps=None):
        """
        recent_history_matrix : [N, 3]，依時間排序的 [load_kw, temperature, ghi]。
                                若是含 load_kw / temperature / ghi 欄名的 DataFrame，會自動依正確順序排列。
        latest_data_timestamp : 最後一筆歷史資料的時間（預測時間戳 = 此時間 + 15 分鐘 x 步數）
        forecast_days         : 預測天數
        future_weather_matrix : (選用) [>= (區塊數-1) x B, 2] 的未來 [temperature, ghi]。
                                只有預測超過 1 個區塊（例如 2 天）時才會用到；
                                未提供時，以歷史最後一天的天氣循環代替。
        history_timestamps    : (選用) 與歷史資料等長的時間序列，提供時會檢查間隔是否為 15 分鐘。
        """
        self.stats["total_requests"] += 1

        # ===== 預測天數 =====
        if (isinstance(forecast_days, bool)
                or not isinstance(forecast_days, (int, float, np.integer, np.floating))
                or not math.isfinite(forecast_days)
                or not 0 < forecast_days <= MAX_FORECAST_DAYS):
            return self._trigger_fallback(
                "horizon_error", f"forecast_days 必須是 0 ~ {MAX_FORECAST_DAYS} 之間的正數")
        total_steps = int(round(forecast_days * STEPS_PER_DAY))
        if total_steps < 1:
            return self._trigger_fallback("horizon_error", "forecast_days 太小，換算後不足 1 步")
        horizon_min = total_steps * INTERVAL_MIN

        # ===== 時間戳 =====
        if latest_data_timestamp is None:
            return self._trigger_fallback("timestamp_error", "latest_data_timestamp 不得為 None",
                                          total_steps=total_steps, horizon_min=horizon_min)
        try:
            parsed_dt = pd.Timestamp(latest_data_timestamp)
            if pd.isna(parsed_dt):
                return self._trigger_fallback("timestamp_error", "無效的時間戳 (NaT)",
                                              total_steps=total_steps, horizon_min=horizon_min)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.tz_localize('Asia/Taipei')
        except Exception as e:
            return self._trigger_fallback("timestamp_error", f"時間戳解析錯誤: {e}",
                                          total_steps=total_steps, horizon_min=horizon_min)
        ctx = dict(parsed_dt=parsed_dt, total_steps=total_steps, horizon_min=horizon_min)

        # ===== 輸入矩陣 =====
        try:
            src = recent_history_matrix
            if isinstance(src, pd.DataFrame):
                if set(FEATURE_ORDER).issubset(src.columns):
                    src = src[FEATURE_ORDER]
                elif src.shape[1] != N_FEATURES:
                    return self._trigger_fallback(
                        "dimension_error", f"DataFrame 需有 {FEATURE_ORDER} 欄位或剛好 3 欄", **ctx)
            arr = np.array(src, dtype=np.float64)
        except Exception:
            return self._trigger_fallback("dimension_error", "輸入歷史資料包含非數值內容", **ctx)

        if arr.ndim != 2 or arr.shape[1] != N_FEATURES:
            return self._trigger_fallback(
                "dimension_error", f"輸入矩陣維度必須為 [N, {N_FEATURES}] (目前為 {arr.shape})", **ctx)

        L = self.lookback_steps
        if len(arr) < L:
            return self._trigger_fallback(
                "length_insufficient", f"歷史資料長度不足，至少需要 {L} 筆，目前 {len(arr)} 筆",
                history_load=arr[:, 0], **ctx)

        window = arr[-L:]
        if not np.isfinite(window).all():
            return self._trigger_fallback("invalid_input", "輸入資料包含 NaN 或 Inf",
                                          history_load=arr[:, 0], **ctx)
        if (window[:, 0] < 0).any():
            return self._trigger_fallback("negative_load_blocked", "輸入歷史負載包含負值",
                                          history_load=arr[:, 0], **ctx)
        if ((window[:, 1] < TEMP_RANGE[0]) | (window[:, 1] > TEMP_RANGE[1])).any():
            return self._trigger_fallback(
                "weather_error", f"歷史氣溫超出合理範圍 {TEMP_RANGE} °C", history_load=arr[:, 0], **ctx)
        if ((window[:, 2] < GHI_RANGE[0]) | (window[:, 2] > GHI_RANGE[1])).any():
            return self._trigger_fallback(
                "weather_error", f"歷史 GHI 超出合理範圍 {GHI_RANGE}", history_load=arr[:, 0], **ctx)

        # ===== 時間間隔（選用）=====
        if history_timestamps is not None:
            try:
                ts = pd.to_datetime(pd.Series(list(history_timestamps)))
                if len(ts) != len(arr):
                    return self._trigger_fallback(
                        "dimension_error",
                        f"history_timestamps 長度 ({len(ts)}) 與歷史資料長度 ({len(arr)}) 不符", **ctx)
                diffs = ts.iloc[-L:].diff().dropna()
                bad = diffs[diffs != pd.Timedelta(minutes=INTERVAL_MIN)]
                if len(bad) > 0:
                    return self._trigger_fallback(
                        "interval_error",
                        f"資料間隔不是 {INTERVAL_MIN} 分鐘，有 {len(bad)} 處不符 (例如 {bad.iloc[0]})", **ctx)
            except Exception as e:
                return self._trigger_fallback("interval_error", f"時間間隔檢查失敗: {e}", **ctx)

        # ===== 未來天氣（只用於第 2 個區塊之後）=====
        B = self.block_steps
        n_blocks = int(math.ceil(total_steps / B))
        need_weather = (n_blocks - 1) * B
        if future_weather_matrix is not None:
            try:
                fw = future_weather_matrix
                if isinstance(fw, pd.DataFrame) and {"temperature", "ghi"}.issubset(fw.columns):
                    fw = fw[["temperature", "ghi"]]
                f_weather = np.array(fw, dtype=np.float64)
            except Exception:
                return self._trigger_fallback("weather_error", "未來天氣包含非數值內容",
                                              history_load=arr[:, 0], **ctx)
            if f_weather.ndim != 2 or f_weather.shape[1] != 2:
                return self._trigger_fallback(
                    "weather_error", f"未來天氣必須為 [步數, 2] (temperature, ghi)，目前 {f_weather.shape}",
                    history_load=arr[:, 0], **ctx)
            if not np.isfinite(f_weather).all() \
                    or ((f_weather[:, 0] < TEMP_RANGE[0]) | (f_weather[:, 0] > TEMP_RANGE[1])).any() \
                    or ((f_weather[:, 1] < GHI_RANGE[0]) | (f_weather[:, 1] > GHI_RANGE[1])).any():
                return self._trigger_fallback("weather_error", "未來天氣含 NaN 或超出合理範圍",
                                              history_load=arr[:, 0], **ctx)
            if len(f_weather) < need_weather:
                return self._trigger_fallback(
                    "weather_error", f"未來天氣至少需要 {need_weather} 筆，目前 {len(f_weather)} 筆",
                    history_load=arr[:, 0], **ctx)
        else:
            reps = int(math.ceil(max(need_weather, 1) / STEPS_PER_DAY))
            f_weather = np.tile(arr[-STEPS_PER_DAY:, 1:], (reps, 1))

        # ===== 模型就緒 =====
        if not self.is_ready:
            return self._trigger_fallback("model_not_ready", self.init_error or "模型未就緒",
                                          history_load=arr[:, 0], **ctx)

        # ===== 推論：每個區塊只預測一次，取全部輸出 =====
        try:
            cur = window.copy()
            blocks = []
            for b in range(n_blocks):
                scaled = self.scaler.transform(cur)
                scaled_pred = np.asarray(
                    self.model.predict(scaled.reshape(1, L, N_FEATURES), verbose=0))
                if scaled_pred.ndim != 2 or scaled_pred.shape[1] != B:
                    return self._trigger_fallback(
                        "model_output_incomplete",
                        f"模型輸出形狀應為 (1, {B})，實際為 {scaled_pred.shape}",
                        history_load=arr[:, 0], **ctx)
                if not np.isfinite(scaled_pred).all():
                    return self._trigger_fallback("model_output_nan", "模型輸出含 NaN/Inf",
                                                  history_load=arr[:, 0], **ctx)
                load_block = self._inverse_load(scaled_pred[0])
                if not np.isfinite(load_block).all():
                    return self._trigger_fallback("model_output_nan", "反標準化後含 NaN/Inf",
                                                  history_load=arr[:, 0], **ctx)
                if (load_block < 0).any():
                    return self._trigger_fallback(
                        "negative_output_blocked",
                        f"模型輸出含負值負載 (最小值 {load_block.min():.2f} kW)",
                        history_load=arr[:, 0], **ctx)
                if self.load_upper is not None and (load_block > self.load_upper).any():
                    return self._trigger_fallback(
                        "model_output_out_of_range",
                        f"模型輸出超過訓練最大值的 {LOAD_UPPER_FACTOR:g} 倍 (最大 {load_block.max():.2f} kW)",
                        history_load=arr[:, 0], **ctx)
                blocks.append(load_block)

                if b < n_blocks - 1:   # 準備下一個區塊的輸入視窗
                    wx = f_weather[b * B:(b + 1) * B]
                    new_rows = np.column_stack([load_block, wx])
                    cur = np.vstack([cur, new_rows])[-L:]

            final = np.concatenate(blocks)[:total_steps]
            timestamps = self._make_timestamps(parsed_dt, len(final))
            return {
                "ai_valid": 1,
                "forecast_load_kw": [round(float(v), 2) for v in final],
                "fallback_load_kw": None,
                "prediction_timestamp": timestamps[0],
                "prediction_timestamps": timestamps,
                "prediction_horizon_min": horizon_min,
                "model_version": self.model_version,
                "error_reason": None,
                "fallback_strategy": None
            }
        except Exception as e:
            return self._trigger_fallback("unexpected_exception", f"多變數推論發生例外: {str(e)}",
                                          history_load=arr[:, 0], **ctx)

    # ------------------------------------------------------
    @staticmethod
    def _build_fallback_values(strategy, history_load, steps):
        """historical_matching：重複最近一天；moving_average：最近最多 7 天同時段平均。"""
        if history_load is None or steps is None:
            return None
        h = np.asarray(history_load, dtype=np.float64)
        if h.ndim != 1 or len(h) < STEPS_PER_DAY:
            return None
        last_day = h[-STEPS_PER_DAY:]
        if not np.isfinite(last_day).all() or (last_day < 0).any():
            return None
        if strategy == "moving_average":
            days = min(len(h) // STEPS_PER_DAY, 7)
            block = h[-days * STEPS_PER_DAY:].reshape(days, STEPS_PER_DAY)
            valid = np.isfinite(block) & (block >= 0)
            counts = valid.sum(axis=0)
            sums = np.where(valid, block, 0.0).sum(axis=0)
            profile = np.where(counts > 0, sums / np.maximum(counts, 1), last_day)
        else:
            profile = last_day
        reps = int(math.ceil(steps / STEPS_PER_DAY))
        return np.tile(profile, reps)[:steps]

    def _trigger_fallback(self, error_type, error_msg, parsed_dt=None, total_steps=None,
                          horizon_min=None, history_load=None):
        self.stats["errors"][error_type] += 1
        strategy = ("historical_matching"
                    if error_type in ["length_insufficient", "timestamp_error", "model_not_ready"]
                    else "moving_average")
        values = self._build_fallback_values(strategy, history_load, total_steps)
        label = strategy if values is not None else "none"
        self.stats["fallback_used"][label] += 1

        timestamps = (self._make_timestamps(parsed_dt, total_steps)
                      if parsed_dt is not None and total_steps else None)
        return {
            "ai_valid": 0,
            "forecast_load_kw": None,
            "fallback_load_kw": [round(float(v), 2) for v in values] if values is not None else None,
            "prediction_timestamp": timestamps[0] if timestamps else None,
            "prediction_timestamps": timestamps,
            "prediction_horizon_min": horizon_min if horizon_min is not None else 0,
            "model_version": self.model_version,
            "error_reason": error_msg,
            "fallback_strategy": label
        }

    def get_system_audit_report(self):
        return self.stats


MultivariateEMSPredictor = MultivariateEMSpredictor  # 大小寫別名，兩種寫法都能用


# =========================================================
# 淨負載計算：Net Load (kW) = Gross Load (kW) - PV (kW)
# =========================================================
def compute_net_load(forecast_result, pv_forecast, pv_col="P_PV_kW_DEMO",
                     time_col="IntervalEnding", pv_capacity_kwp=None,
                     per_kwp_col="P_per_kWp_kW", pv_time_offset_min=0, use_fallback=False):
    """
    forecast_result    : forecast_load_multivariate() 的回傳字典
    pv_forecast        : PV 預報 CSV 路徑或 DataFrame
    pv_capacity_kwp    : 給了就用「每 kWp 出力 x 容量」計算 PV；沒給就直接用 pv_col（目前是 DEMO 值）
    pv_time_offset_min : PV 時間要平移幾分鐘才能和負載預測對齊（例如 -15）
    use_fallback       : AI 預測失敗時，是否允許改用備援負載計算（會在警告中標明）

    回傳 (DataFrame, warnings)。淨負載可以是負值（PV 大於用電），不會被截斷。
    """
    warns = []
    if forecast_result.get("ai_valid") == 1:
        gross, source = forecast_result["forecast_load_kw"], "LSTM"
    elif use_fallback and forecast_result.get("fallback_load_kw"):
        gross, source = forecast_result["fallback_load_kw"], "fallback"
        warns.append("總負載使用備援值（非 AI 預測）。")
    else:
        raise ValueError(f"沒有可用的總負載預測：{forecast_result.get('error_reason')}")

    times = pd.DatetimeIndex(pd.to_datetime(forecast_result["prediction_timestamps"]))
    if times.tz is not None:
        times = times.tz_convert("Asia/Taipei").tz_localize(None)

    if isinstance(pv_forecast, (str, Path)):
        pv_df = pd.read_csv(pv_forecast, encoding="utf-8-sig")
    else:
        pv_df = pv_forecast.copy()
    for c in [time_col, (per_kwp_col if pv_capacity_kwp is not None else pv_col)]:
        if c not in pv_df.columns:
            raise ValueError(f"PV 檔案缺少欄位 {c}，現有欄位：{list(pv_df.columns)}")

    pv_times = pd.DatetimeIndex(pd.to_datetime(pv_df[time_col])) + pd.Timedelta(minutes=pv_time_offset_min)
    if pv_capacity_kwp is not None:
        pv_kw = pd.to_numeric(pv_df[per_kwp_col], errors="coerce").to_numpy(float) * float(pv_capacity_kwp)
    else:
        pv_kw = pd.to_numeric(pv_df[pv_col], errors="coerce").to_numpy(float)
        status = " ".join(pv_df["PV_param_status"].astype(str).unique()) if "PV_param_status" in pv_df else ""
        if "DEMO" in status.upper():
            warns.append(f"PV 欄位為示範參數（{status}），並非實際案場容量。")

    pv_series = pd.Series(pv_kw, index=pv_times)
    if pv_series.index.duplicated().any():
        raise ValueError("PV 預報的時間有重複")
    missing = times.difference(pv_series.index)
    if len(missing) > 0:
        raise ValueError(
            f"PV 預報缺少 {len(missing)}/{len(times)} 個時間點。"
            f"負載預測：{times[0]} ~ {times[-1]}；PV：{pv_series.index[0]} ~ {pv_series.index[-1]}。"
            f"常見原因：日期不同，或時間定義差一格 15 分鐘（可用 pv_time_offset_min=-15 或 15 調整）。")
    pv = pv_series.reindex(times).to_numpy(float)
    if not np.isfinite(pv).all():
        raise ValueError("PV 預報含 NaN")

    gross = np.asarray(gross, dtype=float)
    net = gross - pv
    out = pd.DataFrame({"gross_load_kw": gross, "pv_kw": pv, "net_load_kw": net,
                        "gross_source": source}, index=times)
    out.index.name = "timestamp"

    if gross.mean() > 0 and pv.max() < 0.01 * gross.mean():
        warns.append(f"PV 最大值 {pv.max():.2f} kW，不到平均總負載 {gross.mean():.1f} kW 的 1%，"
                     "淨負載與總負載幾乎相同。")
    n_neg = int((net < 0).sum())
    if n_neg:
        warns.append(f"有 {n_neg} 個時段淨負載為負值（PV 大於用電），代表可充電或反送電。")
    return out, warns