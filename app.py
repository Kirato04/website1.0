# =========================================================
# 微電網 EMS 系統 - 智慧校園用電分析與 AI 預測平台 (app.py)
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

import json
from datetime import datetime
from pathlib import Path

# =========================================================
# 月度用電量預測：計算引擎
# =========================================================
ASSET_DIR = Path(__file__).resolve().parent  # 模型與設定檔存放位置

LOOKBACK_MONTHS = 12


# ---------------------------------------------------------
# 1. 資料整理與 5:1:1 切分
# ---------------------------------------------------------
def build_monthly_series(df, year_col, month_col, value_col, roc_year=True):
    d = df[[year_col, month_col, value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    year = pd.to_numeric(d[year_col], errors="coerce")
    month = pd.to_numeric(d[month_col], errors="coerce")
    if roc_year:
        year = year + 1911
    d["_y"], d["_m"] = year, month
    d = d.dropna(subset=["_y", "_m", value_col])
    if d.empty:
        raise ValueError("沒有可用的「年／月／數值」資料，請確認欄位選擇是否正確")

    idx = pd.to_datetime(
        {"year": d["_y"].astype(int), "month": d["_m"].astype(int), "day": 1}
    )
    s = pd.Series(d[value_col].to_numpy(dtype=float), index=idx).sort_index()

    if s.index.duplicated().any():
        raise ValueError(f"有重複的月份：{list(s.index[s.index.duplicated()].strftime('%Y-%m'))}")
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    missing = full.difference(s.index)
    if len(missing) > 0:
        raise ValueError(f"缺少月份：{list(missing.strftime('%Y-%m'))}")
    if (s < 0).any():
        raise ValueError("資料中含有負值")
    s.index.freq = "MS"
    return s


def split_5_1_1(series):
    n = len(series)
    n_val = n // 7
    n_test = n // 7
    n_train = n - n_val - n_test
    if n_val < 1 or n_train <= LOOKBACK_MONTHS + 1:
        raise ValueError(f"資料只有 {n} 個月，不足以做 5:1:1 切分與 12 個月回看")
    return (
        series.iloc[:n_train],
        series.iloc[n_train:n_train + n_val],
        series.iloc[n_train + n_val:],
    )


# ---------------------------------------------------------
# 2. 標準化縮放與特徵（改用 Z-score 標準化：mean=0, std=1）
# ---------------------------------------------------------
def _fit_scaler(train_values):
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if std < 1e-9:
        std = 1.0
    return {"mean": mean, "std": std}


def _scale(x, sc):
    return (np.asarray(x, dtype=float) - sc["mean"]) / sc["std"]


def _unscale(x, sc):
    return np.asarray(x, dtype=float) * sc["std"] + sc["mean"]


def _month_feats(index):
    ang = 2 * np.pi * np.asarray(index.month) / 12.0
    return np.column_stack([np.sin(ang), np.cos(ang)])


def _window_features(scaled_window, window_index):
    return np.column_stack([scaled_window, _month_feats(window_index)])


def make_samples(scaled, index, start, end, lookback=LOOKBACK_MONTHS):
    X, y = [], []
    for t in range(max(start, lookback), end):
        X.append(_window_features(scaled[t - lookback:t], index[t - lookback:t]))
        y.append(scaled[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------
# 3. 評估指標
# ---------------------------------------------------------
def evaluate(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nz = y_true != 0
    return {
        "MAPE_%": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
    }


# ---------------------------------------------------------
# 4. 模型：季節性基準、Holt-Winters、小型 LSTM
# ---------------------------------------------------------
def seasonal_naive_predict(values, positions):
    return np.array([values[t - 12] for t in positions])


def holt_winters_predict(values, positions):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    out = []
    for t in positions:
        fit = ExponentialSmoothing(
            values[:t], trend="add", seasonal="add",
            seasonal_periods=12, initialization_method="estimated",
        ).fit()
        out.append(float(fit.forecast(1)[0]))
    return np.array(out)


def build_lstm(lookback=LOOKBACK_MONTHS, n_features=3, units=16, dropout=0.2):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, n_features)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_lstm(X_tr, y_tr, X_va, y_va, epochs=500, seed=42):
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = build_lstm(lookback=X_tr.shape[1], n_features=X_tr.shape[2])
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
              batch_size=8, shuffle=True, callbacks=[stop], verbose=0)
    return model


# ---------------------------------------------------------
# 5. 完整實驗流程
# ---------------------------------------------------------
def run_experiment(series, out_dir=".", target_name="度數", unit="kWh",
                   use_holt_winters=True, use_lstm=True, epochs=300, seed=42,
                   save=True):
    n = len(series)
    train, val, test = split_5_1_1(series)
    n_train, n_val = len(train), len(val)
    values, index = series.to_numpy(dtype=float), series.index

    sc = _fit_scaler(train.to_numpy())
    scaled = _scale(values, sc)
    ranges = {"val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}

    preds = {"seasonal_naive": {}}
    for name, (a, b) in ranges.items():
        preds["seasonal_naive"][name] = seasonal_naive_predict(values, range(a, b))

    if use_holt_winters:
        try:
            preds["holt_winters"] = {
                name: holt_winters_predict(values, range(a, b))
                for name, (a, b) in ranges.items()
            }
        except Exception as e:
            print(f"[略過 Holt-Winters] {e}")

    model = None
    if use_lstm:
        try:
            X_tr, y_tr = make_samples(scaled, index, 0, n_train)
            X_va, y_va = make_samples(scaled, index, *ranges["val"])
            model = train_lstm(X_tr, y_tr, X_va, y_va, epochs=epochs, seed=seed)
            preds["lstm"] = {}
            for name, (a, b) in ranges.items():
                X, _ = make_samples(scaled, index, a, b)
                p = model.predict(X, verbose=0).reshape(-1)
                preds["lstm"][name] = _unscale(p, sc)
        except ImportError as e:
            print(f"[略過 LSTM] 沒有安裝 tensorflow：{e}")

    metrics = {
        m: {name: evaluate(values[slice(*ranges[name])], p[name]) for name in ranges}
        for m, p in preds.items()
    }
    best = min(metrics, key=lambda m: metrics[m]["val"]["MAPE_%"])

    frame = pd.DataFrame({"actual": values[n_train:]}, index=index[n_train:])
    frame["split"] = ["val"] * n_val + ["test"] * (n - n_train - n_val)
    for m, p in preds.items():
        frame[m] = np.concatenate([p["val"], p["test"]])

    config = {
        "model_version": "monthly_v1_zscore",
        "lookback_months": LOOKBACK_MONTHS,
        "target_name": target_name,
        "unit": unit,
        "scaler": sc,
        "split_months": {"train": n_train, "val": n_val, "test": n - n_train - n_val},
        "best_model_on_val": best,
        "metrics": metrics,
    }
    if save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "monthly_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if model is not None:
            model.save(str(out / "monthly_model.keras"))

    return {"metrics": metrics, "best_model": best, "predictions": frame, "config": config}


# ---------------------------------------------------------
# 6. 推論與防禦（月版）
# ---------------------------------------------------------
class MonthlyEMSPredictor:
    def __init__(self, model_path="monthly_model.keras", config_path="monthly_config.json"):
        base = Path(__file__).resolve().parent
        self.model_path = base / model_path
        self.config_path = base / config_path
        self.model = None
        self.scaler = None
        self.config = {}
        self.lookback = LOOKBACK_MONTHS
        self.unit = "kWh"
        self.model_version = "unknown"
        self.is_ready = False
        self.init_error = None
        self.stats = {
            "total_requests": 0,
            "errors": {
                "timestamp_error": 0, "horizon_error": 0, "dimension_error": 0,
                "length_insufficient": 0, "invalid_input": 0,
                "negative_value_blocked": 0, "model_not_ready": 0,
                "model_output_invalid": 0, "unexpected_exception": 0,
            },
            "fallback_used": {"seasonal_naive": 0, "none": 0},
        }
        self._load()

    def _load(self):
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"找不到設定檔: {self.config_path}")
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.lookback = int(self.config.get("lookback_months", LOOKBACK_MONTHS))
            self.scaler = self.config["scaler"]
            self.unit = self.config.get("unit", "kWh")
            self.model_version = self.config.get("model_version", "monthly_v1")
            if not self.model_path.exists():
                raise FileNotFoundError(f"找不到模型檔: {self.model_path}")
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self.is_ready = True
        except Exception as e:
            self.is_ready = False
            self.init_error = f"元件載入失敗: {e}"

    @staticmethod
    def _parse_month(x):
        if x is None:
            raise ValueError("latest_month 不得為 None")
        if not isinstance(x, (str, datetime, pd.Timestamp, pd.Period)):
            raise ValueError(f"不支援的型別 {type(x)}")
        ts = x.to_timestamp() if isinstance(x, pd.Period) else pd.Timestamp(x)
        if pd.isna(ts):
            raise ValueError("無效的月份 (NaT)")
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.to_period("M").to_timestamp()

    def forecast_monthly(self, recent_history, latest_month, forecast_months=1):
        self.stats["total_requests"] += 1

        try:
            latest = self._parse_month(latest_month)
        except Exception as e:
            return self._fail("timestamp_error", f"latest_month 無效：{e}", None, None)

        if (isinstance(forecast_months, bool)
                or not isinstance(forecast_months, (int, np.integer))
                or not 1 <= forecast_months <= 12):
            return self._fail("horizon_error", "forecast_months 必須是 1~12 的整數", None, None)
        months = [latest + pd.DateOffset(months=i) for i in range(1, int(forecast_months) + 1)]

        if not isinstance(recent_history, (list, tuple, np.ndarray, pd.Series)):
            return self._fail("dimension_error", "recent_history 必須為 list 或 array", months, None)
        try:
            arr = np.array(recent_history, dtype=np.float64)
        except (ValueError, TypeError):
            return self._fail("dimension_error", "輸入含非數值內容", months, None)
        if arr.ndim != 1:
            return self._fail("dimension_error", f"必須為 1D，目前為 {arr.ndim}D", months, None)

        if len(arr) < self.lookback:
            return self._fail("length_insufficient",
                              f"至少需要 {self.lookback} 個月，目前 {len(arr)} 個月", months, None)

        window = arr[-self.lookback:]
        if not np.isfinite(window).all():
            return self._fail("invalid_input", "輸入含 NaN 或 Inf", months, None)
        if (window < 0).any():
            return self._fail("negative_value_blocked", "輸入含負值", months, None)

        if not self.is_ready:
            return self._fail("model_not_ready", self.init_error or "模型未就緒", months, window)

        try:
            # 標準化下的合理邊界判定（避免超出訓練平均值的 ± 5個標準差範圍）
            upper = self.scaler["mean"] + self.scaler["std"] * 6
            win = list(window)
            preds = []
            for i in range(len(months)):
                end = latest + pd.DateOffset(months=i)
                w_idx = pd.date_range(end=end, periods=self.lookback, freq="MS")
                x = _window_features(_scale(win, self.scaler), w_idx)[None, ...]
                p_scaled = float(np.asarray(
                    self.model.predict(x.astype(np.float32), verbose=0)).reshape(-1)[0])
                p = float(_unscale(p_scaled, self.scaler))
                if not np.isfinite(p) or p < 0 or p > upper:
                    return self._fail("model_output_invalid",
                                      f"第 {i + 1} 個月預測值異常：{p}", months, window)
                preds.append(p)
                win = win[1:] + [p]
        except Exception as e:
            return self._fail("unexpected_exception", f"推論例外：{e}", months, window)

        return {
            "ai_valid": 1,
            "forecast_values": [round(v, 2) for v in preds],
            "fallback_values": None,
            "prediction_months": [m.strftime("%Y-%m") for m in months],
            "unit": self.unit,
            "model_version": self.model_version,
            "error_reason": None,
            "fallback_strategy": None,
        }

    def _fail(self, error_type, msg, months, history):
        self.stats["errors"][error_type] += 1
        fallback, strategy = None, "none"
        if history is not None and months:
            last12 = np.asarray(history)[-12:]
            fallback = [round(float(last12[i % 12]), 2) for i in range(len(months))]
            strategy = "seasonal_naive"
        self.stats["fallback_used"][strategy] += 1
        return {
            "ai_valid": 0,
            "forecast_values": None,
            "fallback_values": fallback,
            "prediction_months": [m.strftime("%Y-%m") for m in months] if months else None,
            "unit": self.unit,
            "model_version": self.model_version,
            "error_reason": msg,
            "fallback_strategy": strategy,
        }

    def get_system_audit_report(self):
        return self.stats


_global_predictor = None


def forecast_monthly(recent_history, latest_month, forecast_months=1):
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = MonthlyEMSPredictor()
    return _global_predictor.forecast_monthly(recent_history, latest_month, forecast_months)


# =========================================================
# 月度用電量預測：網頁畫面
# =========================================================
CAMPUS_SHEET = "107-113各月度數費用合併"
CAMPUS_YEARS = {2018: "107年", 2019: "108年", 2020: "109年", 2021: "110年",
                2022: "111年", 2023: "112年", 2024: "113年"}
CAMPUS_LOCATIONS = ["不分區 (全校總用電)", "行政區", "商學", "校友", "男", "女", "中科", "文創"]

_UNIT_EN = {"度": "kWh", "度數": "kWh", "元": "NTD"}


def _en_unit(u):
    u = (u or "").strip()
    return _UNIT_EN.get(u, u if (u and u.isascii()) else "unit")


def _campus_year_row(df_excel, target_mco, location):
    if location == "不分區 (全校總用電)":
        # 例如 target_mco = "107年" → 找到「107度數」區塊，取其中的「總和」列。
        # （「107費用」區塊也有「總和」列，但那是電費，不能拿來加總）
        label = target_mco.replace("年", "度數")
        n = len(df_excel)
        for idx in range(n):
            v = df_excel.iloc[idx, 0]
            if pd.notna(v) and str(v).strip() == label:
                for j in range(idx + 1, min(idx + 15, n)):
                    w = df_excel.iloc[j, 0]
                    if pd.notna(w) and str(w).strip() == "總和":
                        vals = pd.to_numeric(
                            pd.Series(df_excel.iloc[j, 1:13].values), errors="coerce"
                        ).to_numpy(dtype=float)
                        vals[vals == 0] = np.nan   # 尚未發生的月份（例如 113 年 11、12 月）在 Excel 裡是 0
                        return vals
        return None

    block = None
    for idx, row in df_excel.iterrows():
        val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if target_mco in val and "度數" not in val and "費用" not in val:
            block = target_mco
            continue
        elif any(y in val for y in CAMPUS_YEARS.values()) and val != target_mco:
            block = None
        if block == target_mco and val == location:
            return row.iloc[1:13].values
    return None


def load_campus_series(xl, sheet, location):
    df_excel = xl.parse(sheet, header=None)
    rows, missing = [], []
    for year, mco in CAMPUS_YEARS.items():
        data = _campus_year_row(df_excel, mco, location)
        if data is None:
            missing.append(mco)
            continue
        vals = pd.to_numeric(pd.Series(data), errors="coerce").to_numpy()
        rows.extend((year, m, v) for m, v in enumerate(vals, start=1))
    if missing:
        raise ValueError(f"在檔案中找不到 {'、'.join(missing)} 的【{location}】資料")
    tbl = pd.DataFrame(rows, columns=["year", "month", "kwh"])
    return build_monthly_series(tbl, "year", "month", "kwh", roc_year=False)


def _read_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(file)
        sheet = st.selectbox("工作表", xl.sheet_names, key="monthly_custom_sheet")
        header = st.number_input("標題列位置（第一列為 0）", 0, 50, 0, key="monthly_custom_header")
        return xl.parse(sheet, header=int(header))
    try:
        return pd.read_csv(file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="cp950")


def _guess(cols, keys, default):
    for i, c in enumerate(cols):
        if any(k in str(c) for k in keys):
            return i
    return min(default, len(cols) - 1)


def _clean_digits(df, cols):
    d = df.copy()
    for c in cols:
        d[c] = d[c].astype(str).str.extract(r"(\d+)")[0]
    return d


def _metrics_table(metrics):
    rows = []
    for model, by_split in metrics.items():
        for split, m in by_split.items():
            rows.append({"模型": model, "資料集": split, "MAPE (%)": round(m["MAPE_%"], 2),
                         "RMSE": round(m["RMSE"], 2), "MAE": round(m["MAE"], 2)})
    return pd.DataFrame(rows)


def _plot_predictions(frame, unit):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame.index, frame["actual"], color="black", marker="o", label="Actual")
    colors = {"seasonal_naive": "tab:orange", "holt_winters": "tab:green", "lstm": "tab:red"}
    for m in [c for c in frame.columns if c in colors]:
        ax.plot(frame.index, frame[m], color=colors[m], marker="x", linestyle="--", label=m)
    test_start = frame.index[frame["split"] == "test"][0]
    ax.axvline(test_start, color="gray", linestyle=":", label="Test start")
    ax.set_title("Monthly Forecast Result (Validation & Test)")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Monthly value ({_en_unit(unit)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _plot_monthly_trend(df_target, year, location):
    """月趨勢圖：優先用 Plotly（可互動、中文不會變方塊）；沒安裝 plotly 時退回 Matplotlib。"""
    s = df_target["load_kWh"]
    peak_month, peak = s.idxmax(), float(s.max())
    try:
        import plotly.graph_objects as go
    except ImportError:
        go = None

    if go is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines+markers", name=f"{location} Load",
            line=dict(color="forestgreen", width=3), marker=dict(size=9)))
        fig.add_annotation(
            x=peak_month, y=peak, text=f"Peak: {peak:,.0f} kWh", showarrow=True, arrowhead=2,
            ax=0, ay=-40, font=dict(color="crimson", size=13))
        fig.update_layout(
            title=f"{year} {location} Monthly Electricity Load Analysis",
            xaxis=dict(title=f"Month ({year})", tickmode="array", tickvals=list(s.index),
                       ticktext=[m.strftime("%b") for m in s.index]),
            yaxis_title="Power Consumption (kWh / Month)",
            template="plotly_white", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s.index, s.values, marker="o", color="forestgreen", linewidth=2.5, markersize=7,
            label=f"{location} Load")
    ax.annotate(f"Peak: {peak:,.0f} kWh", xy=(peak_month, peak), xytext=(0, 15),
                textcoords="offset points", ha="center", fontweight="bold", color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
    ax.set_title(f"{year} {location} Monthly Electricity Load Analysis", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Month ({year})", fontsize=12)
    ax.set_ylabel("Power Consumption (kWh / Month)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xticks(s.index)
    ax.set_xticklabels([m.strftime("%b") for m in s.index])
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)


def _plot_forecast(series, months, values, label, unit):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = pd.to_datetime([m + "-01" for m in months])
    
    prev_x = []
    prev_vals = []
    for m in months:
        dt = pd.to_datetime(m + "-01") - pd.DateOffset(years=1)
        if dt in series.index:
            prev_x.append(pd.to_datetime(m + "-01"))
            prev_vals.append(float(series.loc[dt]))

    if prev_vals:
        ax.plot(prev_x, prev_vals, color="tab:blue", marker="s", linestyle="-", label="Actual (Last Year)")

    ax.plot(x, values, color="tab:red", marker="o", linestyle="--", label=label)
    ax.set_title("Monthly Forecast Result")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Monthly value ({_en_unit(unit)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _source_campus():
    file = st.file_uploader("請上傳校園電度電費 Excel 統計檔案",
                            type=["xlsx", "xls"], key="monthly_campus_excel")
    if file is None:
        st.info("💡 請先上傳 107～113 年的電度電費 Excel，系統會自動取出各年 12 個月的用電度數，共 84 個月。")
        return None
    try:
        xl = pd.ExcelFile(file)
        default = CAMPUS_SHEET if CAMPUS_SHEET in xl.sheet_names else xl.sheet_names[0]
        sheet = st.selectbox("資料工作表 (Sheet)", xl.sheet_names,
                             index=xl.sheet_names.index(default), key="monthly_campus_sheet")
        location = st.selectbox("分析地點 / 區域 / 大樓", CAMPUS_LOCATIONS, key="monthly_campus_loc")
        series = load_campus_series(xl, sheet, location)
    except Exception as e:
        st.error(f"❌ 讀取校園 Excel 失敗：{e}")
        return None
    return series, f"度數（{location}）", "kWh", ("campus", file.name, sheet, location)


def _source_custom():
    file = st.file_uploader("上傳月度資料（CSV 或 Excel）", type=["csv", "xlsx", "xls"],
                            key="monthly_upload")
    if file is None:
        st.info("資料需要有「年」「月」「數值（度數或電費）」三種欄位，每個月一列。")
        return None
    try:
        df = _read_upload(file)
    except Exception as e:
        st.error(f"讀取檔案失敗：{e}")
        return None
    st.dataframe(df.head(10), use_container_width=True)

    cols = [str(c) for c in df.columns]
    df.columns = cols
    c1, c2, c3 = st.columns(3)
    ycol = c1.selectbox("年份欄位", cols, index=_guess(cols, ["年"], 0))
    mcol = c2.selectbox("月份欄位", cols, index=_guess(cols, ["月"], 1))
    vcol = c3.selectbox("數值欄位（度數或電費）", cols,
                        index=_guess(cols, ["度數", "電費", "kWh"], 2))
    c4, c5 = st.columns(2)
    roc = c4.radio("年份格式", ["民國年（107）", "西元年（2018）"], horizontal=True).startswith("民國")
    unit = c5.text_input("單位", value="kWh")
    try:
        series = build_monthly_series(_clean_digits(df, [ycol, mcol]), ycol, mcol, vcol, roc_year=roc)
    except Exception as e:
        st.error(f"資料整理失敗：{e}")
        return None
    return series, vcol, unit, ("custom", file.name, ycol, mcol, vcol)


def _pipeline(series, label, unit, key):
    try:
        train, val, test = split_5_1_1(series)
    except Exception as e:
        st.error(f"資料切分失敗：{e}")
        return

    st.success(
        f"共 {len(series)} 個月（{series.index[0]:%Y-%m} ～ {series.index[-1]:%Y-%m}） "
        f"訓練 {len(train)}／驗證 {len(val)}／測試 {len(test)} 個月"
    )

    st.subheader("訓練與評估")
    try:
        import statsmodels  # noqa: F401
        has_hw = True
    except ImportError:
        has_hw = False
    use_hw = st.checkbox("加入 Holt-Winters 比較", value=has_hw, disabled=not has_hw)
    if not has_hw:
        st.caption("未安裝 statsmodels，已略過 Holt-Winters。")
    
    epochs = 300  # 固定最佳預設輪數，無須使用者手動調整

    if st.button("開始訓練並評估", type="primary"):
        with st.spinner("訓練與預測中（Z-score 標準化），請稍候…"):
            try:
                res = run_experiment(series, out_dir=str(ASSET_DIR), target_name=label,
                                        unit=unit, use_holt_winters=use_hw, epochs=epochs)
                
                # 同時自動執行未來 12 個月預測
                predictor = MonthlyEMSPredictor()
                out = predictor.forecast_monthly(series.to_numpy(), series.index[-1], 12)
                
                st.session_state["monthly_result"] = {
                    "res": res, 
                    "forecast_out": out, 
                    "audit_report": predictor.get_system_audit_report(),
                    "key": key
                }
            except Exception as e:
                st.error(f"訓練或預測失敗：{e}")

    saved = st.session_state.get("monthly_result")
    if not (saved and saved["key"] == key):
        return
    res = saved["res"]
    st.write(f"驗證集表現最佳的模型：**{res['best_model']}**")
    with st.expander("📋 點此展開／收合模型評估指標表 (Metrics)", expanded=False):
        st.dataframe(_metrics_table(res["metrics"]), use_container_width=True, hide_index=True)
    st.pyplot(_plot_predictions(res["predictions"], unit))
    st.caption("LSTM 的驗證集同時用於提早停止，所以驗證集成績略偏樂觀；測試集才是最終成績。")

    st.subheader("預測未來月份（未來 12 個月）")
    out = saved.get("forecast_out")
    if out is not None:
        if out["ai_valid"] == 1:
            vals, line_label = out["forecast_values"], "LSTM forecast"
            st.success("LSTM 預測成功")
        else:
            vals, line_label = out["fallback_values"], "Fallback (same month last year)"
            st.warning(f"預測無法使用：{out['error_reason']}")
        
        if vals is None:
            st.error("也無法產生備援值。")
        else:
            tbl = pd.DataFrame({"月份": out["prediction_months"], f"預測值 ({unit})": vals})
            with st.expander("📋 點此展開／收合預測數據表格", expanded=False):
                st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.pyplot(_plot_forecast(series, out["prediction_months"], vals, line_label, unit))
            st.download_button("下載預測結果 CSV", tbl.to_csv(index=False).encode("utf-8-sig"),
                               "monthly_forecast.csv", "text/csv")
        
        with st.expander("系統稽核報表"):
            st.json(saved.get("audit_report", {}))


def render_monthly_tab():
    source = st.radio("資料來源", ["校園電度電費 Excel（107～113 年）", "自訂檔案（CSV／Excel，年月數值欄位）"],
                      horizontal=True, key="monthly_source")
    got = _source_campus() if source.startswith("校園") else _source_custom()
    if got is None:
        return
    _pipeline(*got)


st.set_page_config(page_title="微電網 AI 預測與分析系統", layout="wide")

# 💡 引入 Google Fonts (Noto Sans TC 思源黑體) 並套用到全站
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans TC', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ 微電網 EMS 智慧用電分析與 AI 預測平台")
st.markdown("結合**校園長短期用電統計分析**與 **月度用電量 AI 預測**。")

tab1, tab2 = st.tabs([
    "📈 1. 歷史用電統計與動態趨勢分析", 
    "🚀 2. 月度用電量 AI 預測系統"
])

with tab1:
    st.header("📈 歷史用電統計與動態趨勢分析")
    
    with st.expander("📌 點此查看【歷史統計檔案】的規格與欄位說明", expanded=False):
        st.markdown("""
        * **檔案格式**：Excel 試算表（`.xlsx` 或 `.xls`）。
        * **工作表名稱（Sheet）**：需包含各月份用電度數彙整表（例如 `107-113各月度數費用合併`）。
        """)

    uploaded_excel_tab1 = st.file_uploader(
        "請上傳校園電度電費 Excel 統計檔案", 
        type=["xlsx", "xls"], 
        key="excel_uploader_tab1"
    )

    if uploaded_excel_tab1 is not None:
        try:
            xls = pd.ExcelFile(uploaded_excel_tab1)
            default_sheet = '107-113各月度數費用合併' if '107-113各月度數費用合併' in xls.sheet_names else xls.sheet_names[0]
            sheet_to_use = st.selectbox("請選擇資料工作表 (Sheet)", xls.sheet_names, index=xls.sheet_names.index(default_sheet) if default_sheet in xls.sheet_names else 0, key="sheet_tab1")
            
            df_excel = pd.read_excel(uploaded_excel_tab1, sheet_name=sheet_to_use, header=None)
            st.success(f"✅ 成功讀取檔案：{uploaded_excel_tab1.name} (工作表: {sheet_to_use})")

            col_y1, col_y2 = st.columns(2)
            with col_y1:
                selected_year = st.selectbox("選擇分析西元年份", [2018, 2019, 2020, 2021, 2022, 2023, 2024], index=6, key="trend_year")
                mco_year_map = {
                    2018: '107年', 2019: '108年', 2020: '109年', 
                    2021: '110年', 2022: '111年', 2023: '112年', 2024: '113年'
                }
                target_mco = mco_year_map.get(selected_year, '113年')

            with col_y2:
                location_options = ['不分區 (全校總用電)', '行政區', '商學', '校友', '男', '女', '中科', '文創']
                selected_location = st.selectbox("選擇分析地點 / 區域 / 大樓", location_options, key="trend_location")

            target_row_data = _campus_year_row(df_excel, target_mco, selected_location)

            if target_row_data is not None:
                months_range = pd.date_range(start=f'{selected_year}-01-01', end=f'{selected_year}-12-01', freq='MS')
                df_target = pd.DataFrame({'load_kWh': pd.to_numeric(target_row_data, errors='coerce')}, index=months_range)
                df_target = df_target.dropna()

                if len(df_target) > 0:
                    total_load = df_target['load_kWh'].sum()
                    mean_load = df_target['load_kWh'].mean()
                    max_load = df_target['load_kWh'].max()
                    min_load = df_target['load_kWh'].min()
                    max_month = df_target['load_kWh'].idxmax().strftime('%Y-%m')
                    min_month = df_target['load_kWh'].idxmin().strftime('%Y-%m')

                    st.markdown("---")
                    st.subheader(f"📊 {selected_year} 年 ({target_mco}) - 【{selected_location}】 用電統計摘要")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("區間總用電量", f"{total_load:,.2f} kWh")
                    col2.metric("每月平均用電", f"{mean_load:,.2f} kWh")
                    col3.metric("用電最高月份", max_month, f"{max_load:,.2f} kWh")
                    col4.metric("用電最低月份", min_month, f"{min_load:,.2f} kWh")

                    _plot_monthly_trend(df_target, selected_year, selected_location)
                else:
                    st.warning(f"⚠️ {selected_year} 年的【{selected_location}】目前沒有有效的月用電數值。")
            else:
                st.error(f"❌ 在上傳的檔案中找不到 {selected_year} 年 ({target_mco}) 對應的【{selected_location}】資料。")

        except Exception as e:
            st.error(f"❌ 解析上傳的 Excel 檔案時發生錯誤: {str(e)}")
    else:
        st.info("💡 請先上傳您的電度電費 Excel 統計檔案，即可檢視月統計與趨勢圖。")

with tab2:
    st.header("🚀 月度用電量 AI 預測系統")
    render_monthly_tab()