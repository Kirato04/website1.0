# =========================================================
# 微電網 EMS 智慧負載預測與淨負載對接平台（多變數氣象整合版）
# app_multivariate.py
# 模型輸入：[load_kw, temperature, ghi]，回看 L 步 -> 一次輸出 B 步總負載
# =========================================================
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from infer_multivariate import (
    MultivariateEMSPredictor,
    compute_net_load,
    STEPS_PER_DAY,
    INTERVAL_MIN,
)

BASE_DIR = Path(__file__).resolve().parent

# ★ 這三個檔名請依你 GitHub repo 裡的實際檔名修改
DEFAULT_BUILDING_FILE = "Report251014_m_buildings.csv"   # 歷史大樓用電
DEFAULT_WEATHER_FILE = "Weather_15min.csv"               # 歷史天氣（時間、氣溫、GHI）
PV_GLOB = "PV_Forecast_15min_*.csv"                      # 光伏預報（檔名含日期）

# 欄位名稱自動辨識用的關鍵字（不分大小寫）
TIME_KEYS = ["timestamp", "datetime", "time", "date", "intervalending", "時間", "日期"]
TEMP_KEYS = ["temperature", "temp", "氣溫", "溫度"]
GHI_KEYS = ["ghi", "日射", "輻射"]

st.set_page_config(
    page_title="微電網 EMS 負載預測與調度系統 (多變數氣象整合版)",
    page_icon="⚡",
    layout="wide",
)


# =========================================================
# 模型載入（只載入一次）
# =========================================================
@st.cache_resource
def load_predictor():
    return MultivariateEMSPredictor(
        model_path="lstm_model_multivariate.h5",
        scaler_path="data_scaler_multivariate.pkl",
    )


predictor = load_predictor()


# =========================================================
# 工具函式
# =========================================================
def _read_csv(src):
    """讀 CSV（路徑或上傳檔都可以）。每次都先把游標歸零，避免第二次讀到空檔。"""
    def _rewind():
        if hasattr(src, "seek"):
            src.seek(0)

    _rewind()
    try:
        return pd.read_csv(src, encoding="utf-8-sig")
    except UnicodeDecodeError:
        _rewind()
        return pd.read_csv(src, encoding="cp950")


def _find_col(columns, keys):
    for c in columns:
        name = str(c).strip().lower()
        if any(k in name for k in keys):
            return c
    return None


def _to_naive_time(series):
    t = pd.to_datetime(series, errors="coerce")
    if t.dt.tz is not None:
        t = t.dt.tz_localize(None)
    return t


def _find_time_col(df):
    """先找名稱像時間的欄位，找不到再逐欄試著當日期解析。"""
    c = _find_col(df.columns, TIME_KEYS)
    candidates = ([c] if c is not None else []) + [x for x in df.columns if x != c]
    for col in candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if _to_naive_time(df[col]).notna().mean() > 0.9:
            return col
    return None


def _default_target_date():
    files = sorted(BASE_DIR.glob(PV_GLOB))
    if files:
        m = re.search(r"(\d{8})", files[-1].name)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y%m%d").date()
            except ValueError:
                pass
    return datetime(2026, 10, 19).date()


def _prepare_weather(df):
    """把天氣檔整理成 15 分鐘間隔、以時間為索引、欄位 temperature / ghi 的表。"""
    tcol = _find_time_col(df)
    tc = _find_col(df.columns, TEMP_KEYS)
    gc = _find_col(df.columns, GHI_KEYS)
    if tcol is None or tc is None or gc is None:
        raise ValueError(
            f"天氣檔需要「時間、氣溫、GHI」三種欄位，目前欄位：{list(df.columns)}"
        )
    w = pd.DataFrame(
        {
            "temperature": pd.to_numeric(df[tc], errors="coerce").to_numpy(float),
            "ghi": pd.to_numeric(df[gc], errors="coerce").to_numpy(float),
        },
        index=_to_naive_time(df[tcol]),
    )
    w = w[w.index.notna()].sort_index()
    w = w[~w.index.duplicated()]
    # 若天氣是每小時一筆，這一步會用時間內插補成 15 分鐘
    w = w.resample(f"{INTERVAL_MIN}min").mean().interpolate(method="time", limit=8)
    return w


def build_model_inputs(df_build, df_weather, latest_ts, forecast_days):
    """
    整理成模型要的輸入。回傳 dict：
      matrix      : [L, 3] 的 [load_kw, temperature, ghi]
      timestamps  : 歷史時間（用來檢查 15 分鐘間隔），找不到時間欄則為 None
      future_weather : 預測 2 天以上時才會用到的未來天氣，否則 None
      notes       : [(等級, 訊息), ...] 顯示給使用者看
    """
    L = predictor.lookback_steps
    B = predictor.block_steps
    notes = []

    # ---- 1. 負載欄與時間欄 ----
    tcol = _find_time_col(df_build)
    load_col = "人言大樓kw" if "人言大樓kw" in df_build.columns else None
    if load_col is None:
        for c in df_build.columns:
            if c != tcol and pd.to_numeric(df_build[c], errors="coerce").notna().mean() > 0.9:
                load_col = c
                break
    if load_col is None:
        raise ValueError(f"找不到負載欄位，目前欄位：{list(df_build.columns)}")
    notes.append(("info", f"使用的負載欄位：{load_col}"))

    # ---- 2. 取「目標日前一刻為止」的最後 L 筆 ----
    if tcol is None:
        d = df_build.tail(L).copy()
        idx = None
        if df_weather is not None:
            raise ValueError("大樓 CSV 找不到時間欄，無法和天氣檔依時間對齊。")
        notes.append(("warning", f"大樓 CSV 找不到時間欄，已直接取最後 {L} 筆，"
                                 "無法確認是否剛好接在目標日之前。"))
    else:
        d = (
            df_build.assign(_t=_to_naive_time(df_build[tcol]))
            .dropna(subset=["_t"])
            .sort_values("_t")
            .drop_duplicates("_t")
            .set_index("_t")
        )
        d = d[d.index <= latest_ts]
        if len(d) < L:
            raise ValueError(
                f"目標日之前的歷史資料只有 {len(d)} 筆，模型需要 {L} 筆。"
                "請改選較晚的目標日，或上傳更完整的歷史用電 CSV。"
            )
        d = d.tail(L)
        idx = d.index
        gap = latest_ts - idx[-1]
        if gap > pd.Timedelta(minutes=INTERVAL_MIN):
            notes.append(("warning", f"歷史資料最後一筆是 {idx[-1]}，離目標日起點 "
                                     f"（{latest_ts}）有 {gap} 的空缺，預測品質可能下降。"))

    load = pd.to_numeric(d[load_col], errors="coerce").to_numpy(float)

    # ---- 3. 天氣：大樓 CSV 自帶 > 天氣檔 > 示範假資料 ----
    w = _prepare_weather(df_weather) if df_weather is not None else None
    others = [c for c in d.columns if c != load_col]
    t_in = _find_col(others, TEMP_KEYS)
    g_in = _find_col(others, GHI_KEYS)

    if t_in is not None and g_in is not None:
        temp = pd.to_numeric(d[t_in], errors="coerce").to_numpy(float)
        ghi = pd.to_numeric(d[g_in], errors="coerce").to_numpy(float)
        notes.append(("info", f"天氣來源：大樓 CSV 內的欄位（{t_in}、{g_in}）"))
    elif w is not None:
        hw = w.reindex(idx)
        if hw.isna().any().any():
            raise ValueError(
                f"天氣檔在歷史視窗（{idx[0]} ~ {idx[-1]}）內缺 "
                f"{int(hw.isna().any(axis=1).sum())} 筆，請確認天氣檔涵蓋這段時間。"
            )
        temp, ghi = hw["temperature"].to_numpy(float), hw["ghi"].to_numpy(float)
        notes.append(("info", "天氣來源：天氣檔（已依時間與用電資料對齊）"))
    else:
        temp = 25.0 + 3.0 * np.sin(np.linspace(0, 2 * np.pi, L))
        ghi = np.maximum(0, 800 * np.sin(np.linspace(0, np.pi, L)))
        notes.append(("warning", "⚠️ 沒有提供天氣資料，目前使用的是「示範用假天氣」，"
                                 "預測結果不能當作含天氣模型的真實表現。"
                                 "請上傳天氣 CSV 或把天氣檔放進 repo。"))

    # ---- 4. 預測超過一個區塊（例如 2 天）時，需要未來天氣 ----
    future = None
    n_blocks = int(np.ceil(forecast_days * STEPS_PER_DAY / B))
    need = (n_blocks - 1) * B
    if need > 0:
        if w is not None:
            fidx = pd.date_range(latest_ts + pd.Timedelta(minutes=INTERVAL_MIN),
                                 periods=need, freq=f"{INTERVAL_MIN}min")
            fw = w.reindex(fidx)
            if fw.notna().all().all():
                future = fw[["temperature", "ghi"]].to_numpy(float)
                notes.append(("info", "第 2 天起使用天氣檔裡的未來天氣。"))
        if future is None:
            notes.append(("info", "天氣檔沒有涵蓋未來日期，第 2 天起的天氣以歷史最後一天循環代替。"))

    return {
        "matrix": np.column_stack([load, temp, ghi]),
        "timestamps": idx,
        "future_weather": future,
        "notes": notes,
    }


def load_pv(uploaded_pv, target_date):
    """回傳 (DataFrame, 檔名)。上傳優先；否則在 repo 內找檔名含目標日期的 PV 檔。"""
    if uploaded_pv is not None:
        return _read_csv(uploaded_pv), uploaded_pv.name
    files = sorted(BASE_DIR.glob(PV_GLOB))
    if not files:
        return None, None
    key = target_date.strftime("%Y%m%d")
    pick = [f for f in files if key in f.name]
    f = pick[0] if pick else files[-1]
    return _read_csv(f), f.name


def make_figure(df, with_pv):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["gross_load_kw"], name="Gross Load (預測總負載)",
                             line=dict(color="crimson", width=2.5)))
    if with_pv:
        fig.add_trace(go.Scatter(x=df.index, y=df["pv_kw"], name="PV Forecast (光伏預報)",
                                 line=dict(color="orange", width=2.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["net_load_kw"],
                                 name="Net Load (淨負載 = 總負載 - PV)",
                                 line=dict(color="dodgerblue", width=2.5, dash="dash")))
    fig.update_layout(
        title="微電網 15 分鐘級電力功率流向預測 (多變數氣象整合版)",
        xaxis_title="時間戳記",
        yaxis_title="功率 (kW)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    return fig


def _show_notes(notes):
    for level, msg in notes:
        (st.warning if level == "warning" else st.info)(msg)


# =========================================================
# 頁面標題與模型狀態
# =========================================================
st.title("⚡ 微電網 EMS 智慧負載預測與淨負載對接平台")
st.markdown("---")

if not predictor.is_ready:
    st.error(f"❌ 模型載入失敗: {predictor.init_error}")
else:
    st.success("✅ 多變數 3D LSTM 模型（歷史負載、氣溫、GHI）已成功載入並就緒！")
    st.caption(f"模型回看 {predictor.lookback_steps} 步 → 一次預測 {predictor.block_steps} 步（每步 {INTERVAL_MIN} 分鐘）")

# =========================================================
# 側邊欄
# =========================================================
st.sidebar.header("⚙️ 預測參數設定")
forecast_days = st.sidebar.slider("預測天數 (Days)", min_value=1, max_value=7, value=1)
target_date = st.sidebar.date_input("預測目標起始日", value=_default_target_date())
pv_offset = st.sidebar.number_input(
    "PV 時間平移（分鐘）", value=0, step=15,
    help="若出現「PV 預報缺少時間點」，可試 -15 或 15，用來修正 PV 與負載預測差一格的問題。",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 檔案上傳與來源")
uploaded_building = st.sidebar.file_uploader("上傳歷史大樓用電 CSV", type=["csv"])
uploaded_weather = st.sidebar.file_uploader("上傳歷史天氣 CSV（時間、氣溫、GHI）", type=["csv"])
uploaded_pv = st.sidebar.file_uploader("上傳光伏預報 CSV", type=["csv"])

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 24小時預測與淨負載圖表",
    "📊 批次數據預覽與下載",
    "🧪 系統防禦與稽核統計",
    "📖 說明與規格說明",
])

# =========================================================
# 分頁 1：預測
# =========================================================
with tab1:
    st.subheader("🌐 總負載 (Gross Load) 與淨負載 (Net Load) 即時預測")

    if st.button("🚀 執行多變數智慧預測與對接運算", type="primary"):
        for k in ("gross_df", "net_df", "warns", "forecast_res", "input_notes", "run_error"):
            st.session_state.pop(k, None)

        with st.spinner("正在進行 3 特徵 LSTM 模型推論與淨負載計算..."):
            try:
                # 1. 讀取歷史用電
                if uploaded_building is not None:
                    df_build = _read_csv(uploaded_building)
                else:
                    p = BASE_DIR / DEFAULT_BUILDING_FILE
                    if not p.exists():
                        raise FileNotFoundError(
                            f"repo 裡找不到 {DEFAULT_BUILDING_FILE}，請從左側上傳歷史大樓用電 CSV。")
                    df_build = _read_csv(p)

                # 2. 讀取歷史天氣（可選）
                if uploaded_weather is not None:
                    df_weather = _read_csv(uploaded_weather)
                elif (BASE_DIR / DEFAULT_WEATHER_FILE).exists():
                    df_weather = _read_csv(BASE_DIR / DEFAULT_WEATHER_FILE)
                else:
                    df_weather = None

                # 3. 整理輸入並預測
                latest_ts = (pd.Timestamp(datetime.combine(target_date, datetime.min.time()))
                             - pd.Timedelta(minutes=INTERVAL_MIN))
                inp = build_model_inputs(df_build, df_weather, latest_ts, forecast_days)
                st.session_state["input_notes"] = inp["notes"]

                forecast_res = predictor.forecast_load_multivariate(
                    recent_history_matrix=inp["matrix"],
                    latest_data_timestamp=latest_ts,
                    forecast_days=forecast_days,
                    future_weather_matrix=inp["future_weather"],
                    history_timestamps=inp["timestamps"],
                )
                st.session_state["forecast_res"] = forecast_res

                if forecast_res["ai_valid"] != 1:
                    st.session_state["run_error"] = (
                        f"AI 預測未通過防禦檢查：{forecast_res['error_reason']}")
                else:
                    times = pd.DatetimeIndex(pd.to_datetime(forecast_res["prediction_timestamps"]))
                    if times.tz is not None:
                        times = times.tz_convert("Asia/Taipei").tz_localize(None)
                    st.session_state["gross_df"] = pd.DataFrame(
                        {"gross_load_kw": forecast_res["forecast_load_kw"]}, index=times)

                    # 4. 光伏預報與淨負載
                    pv_df, pv_name = load_pv(uploaded_pv, target_date)
                    if pv_df is None:
                        st.session_state["warns"] = ["找不到光伏預報檔，僅顯示總負載預測。"]
                    else:
                        net_df, warns = compute_net_load(
                            forecast_result=forecast_res,
                            pv_forecast=pv_df,
                            pv_col="P_PV_kW_DEMO" if "P_PV_kW_DEMO" in pv_df.columns else "P_PV_kW",
                            time_col="IntervalEnding" if "IntervalEnding" in pv_df.columns else "timestamp",
                            pv_time_offset_min=int(pv_offset),
                        )
                        st.session_state["net_df"] = net_df
                        st.session_state["warns"] = [f"使用的 PV 檔：{pv_name}"] + warns
            except Exception as e:
                st.session_state["run_error"] = f"運算過程發生錯誤: {e}"

    # ---------- 顯示結果 ----------
    _show_notes(st.session_state.get("input_notes", []))
    if st.session_state.get("run_error"):
        st.error(f"❌ {st.session_state['run_error']}")

    for w in st.session_state.get("warns", []):
        st.warning(f"⚠️ {w}")

    ndf = st.session_state.get("net_df")
    gdf = st.session_state.get("gross_df")
    if ndf is not None:
        st.success("🎉 預測與淨負載計算完成！")
        st.plotly_chart(make_figure(ndf, True), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("平均總負載", f"{ndf['gross_load_kw'].mean():.2f} kW")
        c2.metric("平均光伏發電", f"{ndf['pv_kw'].mean():.2f} kW")
        c3.metric("平均淨負載", f"{ndf['net_load_kw'].mean():.2f} kW")
    elif gdf is not None:
        st.plotly_chart(make_figure(gdf, False), use_container_width=True)
        st.metric("平均總負載", f"{gdf['gross_load_kw'].mean():.2f} kW")

# =========================================================
# 分頁 2：資料與下載
# =========================================================
with tab2:
    st.subheader("📊 預測輸出 CSV 檔案檢視與下載")
    out_df = st.session_state.get("net_df")
    if out_df is None:
        out_df = st.session_state.get("gross_df")
    if out_df is not None:
        st.dataframe(out_df, use_container_width=True)
        st.download_button(
            label="📥 下載 EMS 標準對接輸出檔 (CSV)",
            data=out_df.reset_index().to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"EMS_Output_Multivariate_{target_date}.csv",
            mime="text/csv",
        )
    else:
        st.info("💡 請先至「24小時預測與淨負載圖表」頁籤點擊執行預測按鈕以產生數據。")

# =========================================================
# 分頁 3：稽核
# =========================================================
with tab3:
    st.subheader("🧪 系統防禦模組與稽核統計")
    st.json(predictor.get_system_audit_report())

# =========================================================
# 分頁 4：說明
# =========================================================
with tab4:
    st.subheader("📖 多變數模型說明與規格")
    st.markdown("""
    - **模型架構**：3 特徵輸入 (`load_kw`, `temperature`, `ghi`)，讀取過去一天的資料，一次輸出未來 96 步（24 小時）的總負載預測。
    - **防禦機制**：包含 NaN 檢查、負值攔截、氣溫與 GHI 合理性檢查、時間間隔檢查，以及自動備援機制。
    - **淨負載定義**：淨負載可為負值（代表儲能充電或反送電）。
    """)
    st.latex(r"\text{Net Load} = \text{Gross Load} - \text{PV}")
    st.markdown("""
    - **天氣來源順序**：大樓 CSV 內的天氣欄位 → 天氣 CSV → 示範假天氣（會出現紅色警告）。
    - **預測 2 天以上**：第 2 天起若沒有未來天氣，會用歷史最後一天的天氣循環代替，準確度會較低。
    """)