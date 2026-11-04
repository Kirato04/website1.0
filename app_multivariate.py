# =========================================================
# 微電網 EMS 智慧負載預測與淨負載對接平台（多變數氣象整合版）
# app_multivariate.py
# 模型輸入：[load_kw, temperature, ghi]，回看 L 步 -> 一次輸出 B 步總負載
# =========================================================
import re
from datetime import datetime, timedelta
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
    try:
        t = pd.to_datetime(series, errors="coerce", format="mixed")   # 同一欄可混用 2026/10/19、2026-10-20
    except (ValueError, TypeError):
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


def _clean_pv_frame(df):
    """
    整理單一 PV 檔：時間欄統一叫 IntervalEnding，並轉成真正的日期時間。
    各檔的日期格式不一定相同（2026/10/19、2026-10-20、2026/11/1），所以要逐檔解析。
    """
    out = df.copy()
    if "IntervalEnding" not in out.columns:
        if "timestamp" in out.columns:
            out = out.rename(columns={"timestamp": "IntervalEnding"})
        else:
            raise ValueError(f"PV 檔缺少時間欄 IntervalEnding，目前欄位：{list(out.columns)}")
    out["IntervalEnding"] = _to_naive_time(out["IntervalEnding"])
    return out.dropna(subset=["IntervalEnding"])


def _merge_clean_pv(frames):
    """把多個（已整理過的）PV 檔合併成一張表：依時間排序，同一時間點只留一筆。"""
    if not frames:
        return None
    pv = pd.concat(frames, ignore_index=True)
    return (pv.sort_values("IntervalEnding", kind="stable")
              .drop_duplicates("IntervalEnding", keep="last")
              .reset_index(drop=True))


def pv_coverage(pv):
    """
    PV 預報涵蓋哪些「預報日」。IntervalEnding 從 00:15 排到隔天 00:00 才算同一天，
    所以先減 15 分鐘再取日期；一天不滿 96 筆的日期視為缺資料。
    """
    if pv is None or pv.empty:
        return None
    day = (pv["IntervalEnding"] - pd.Timedelta(minutes=INTERVAL_MIN)).dt.normalize()
    counts = day.value_counts().sort_index()
    full = sorted(d.date() for d in counts[counts >= STEPS_PER_DAY].index)
    if not full:
        return None
    full_set = set(full)
    span = pd.date_range(counts.index[0], counts.index[-1], freq="D")
    return {
        "first": full[0],
        "last": full[-1],
        "n_days": len(full),
        "available": full_set,
        "missing": [d.date() for d in span if d.date() not in full_set],
    }


@st.cache_data(show_spinner=False)
def load_repo_pv():
    """讀取 repo 內所有 PV 預報檔並合併（每天一個檔案也可以）。"""
    frames, names = [], []
    for f in sorted(BASE_DIR.glob(PV_GLOB)):
        try:
            frames.append(_clean_pv_frame(_read_csv(f)))
            names.append(f.name)
        except Exception:
            continue
    return _merge_clean_pv(frames), names


def get_pv_all(uploaded_files):
    """回傳 (合併後的 PV 表, 檔名清單, 錯誤訊息清單)。有上傳檔案時以上傳的為準。"""
    if not uploaded_files:
        pv, names = load_repo_pv()
        return pv, names, []
    frames, names, errs = [], [], []
    for f in uploaded_files:
        try:
            frames.append(_clean_pv_frame(_read_csv(f)))
            names.append(f.name)
        except Exception as e:
            errs.append(f"{f.name}：{e}")
    return _merge_clean_pv(frames), names, errs


def pv_ghi_series(pv_df, interval_ending):
    """
    由 PV 預報檔反推 GHI（W/m²）。
    每 kWp 出力 = PR × GHI / 1000，所以 GHI = 每 kWp 出力 ÷ PR × 1000。
    回傳的索引使用「用電資料的時間標記」；欄位不足時回傳 None。
    """
    if pv_df is None or "P_per_kWp_kW" not in pv_df.columns:
        return None
    pr = 0.8
    if "PV_param_status" in pv_df.columns:
        m = re.search(r"PR\s*=\s*([0-9.]+)", " ".join(pv_df["PV_param_status"].astype(str).unique()))
        if m:
            try:
                pr = float(m.group(1))
            except ValueError:
                pass
    if pr <= 0:
        return None
    ghi = pd.to_numeric(pv_df["P_per_kWp_kW"], errors="coerce").to_numpy(float) / pr * 1000.0
    idx = pd.DatetimeIndex(pv_df["IntervalEnding"])
    if not interval_ending:
        idx = idx - pd.Timedelta(minutes=INTERVAL_MIN)
    s = pd.Series(ghi, index=idx).clip(lower=0.0, upper=1500.0)
    return s[~s.index.duplicated()]


def _ghi_by_time_of_day(ghi_s, idx, ref_day, interval_ending):
    """拿「某一天」的 GHI 曲線，依時刻（幾點幾分）套到 idx 的每個時間點上。"""
    lo = pd.Timestamp(ref_day).normalize()
    hi = lo + pd.Timedelta(days=1)
    if interval_ending:
        sl = ghi_s[(ghi_s.index > lo) & (ghi_s.index <= hi)]
    else:
        sl = ghi_s[(ghi_s.index >= lo) & (ghi_s.index < hi)]
    if len(sl) < STEPS_PER_DAY:
        return None
    tod = pd.Series(sl.to_numpy(float), index=[(t.hour * 60 + t.minute) % 1440 for t in sl.index])
    vals = tod.reindex([(t.hour * 60 + t.minute) % 1440 for t in idx]).to_numpy(float)
    return vals if np.isfinite(vals).all() else None


def _demo_temp(n):
    return 25.0 + 3.0 * np.sin(np.linspace(0, 2 * np.pi, n))


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


def build_model_inputs(df_build, df_weather, latest_ts, forecast_days, target_day,
                       pv_df=None, interval_ending=True, allow_demo=True):
    """
    整理成模型要的輸入。回傳 dict：
      matrix         : [L, 3] 的 [load_kw, temperature, ghi]
      timestamps     : 歷史時間（用來檢查 15 分鐘間隔），找不到時間欄則為 None
      future_weather : 預測 2 天以上時才會用到的未來天氣，否則 None
      demo           : 是否為示範模式（用電資料沒有涵蓋目標日）
      notes          : [(等級, 訊息), ...] 顯示給使用者看
    """
    L = predictor.lookback_steps
    B = predictor.block_steps
    notes = []
    demo = False
    target_day = pd.Timestamp(target_day).normalize()

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

    # ---- 2. 取歷史視窗：優先「目標日前一刻為止的最後 L 筆」；沒有涵蓋時走示範模式 ----
    if tcol is None:
        d = df_build.tail(L).copy()
        idx = None
        if df_weather is not None:
            raise ValueError("大樓 CSV 找不到時間欄，無法和天氣檔依時間對齊。")
        notes.append(("warning", f"大樓 CSV 找不到時間欄，已直接取最後 {L} 筆，"
                                 "無法確認是否剛好接在目標日之前。"))
    else:
        all_d = (
            df_build.assign(_t=_to_naive_time(df_build[tcol]))
            .dropna(subset=["_t"])
            .sort_values("_t")
            .drop_duplicates("_t")
            .set_index("_t")
        )
        d = all_d[all_d.index <= latest_ts]
        gap = (latest_ts - d.index[-1]) if len(d) else None
        too_short = len(d) < L
        too_far = (not too_short) and gap > pd.Timedelta(days=1)

        if too_short or too_far:
            if not allow_demo:
                if too_short:
                    raise ValueError(
                        f"目標日之前的歷史資料只有 {len(d)} 筆，模型需要 {L} 筆。"
                        "可勾選側邊欄的「示範模式」，或改選較晚的目標日／上傳涵蓋該日期的用電 CSV。")
                raise ValueError(
                    f"用電 CSV 最後一筆是 {d.index[-1]}，離目標日起點（{latest_ts}）相差 {gap}，"
                    "沒有涵蓋目標日前一天。可勾選側邊欄的「示範模式」。")
            if len(all_d) < L:
                raise ValueError(f"用電 CSV 總共只有 {len(all_d)} 筆，少於模型需要的 {L} 筆。")
            d = all_d.tail(L)
            idx = d.index
            demo = True
            notes.append(("warning",
                          f"⚠️ 示範模式：用電 CSV 的資料是 {idx[0]} ~ {idx[-1]}，沒有涵蓋目標日 "
                          f"{target_day.date()}。模型輸入使用這最後 {L} 筆，預測結果標示為目標日，"
                          "並對上同一天的 PV 預報。因此每個目標日的「負載輸入」都相同，"
                          "只有天氣會不同，這不是該日的真實預測績效。"))
        else:
            d = d.tail(L)
            idx = d.index
            if gap > pd.Timedelta(minutes=INTERVAL_MIN):
                notes.append(("warning", f"歷史資料最後一筆是 {idx[-1]}，離目標日起點 "
                                         f"（{latest_ts}）有 {gap} 的空缺，預測品質可能下降。"))

    load = pd.to_numeric(d[load_col], errors="coerce").to_numpy(float)

    # ---- 3. 天氣：大樓 CSV 自帶 > 天氣檔 > PV 預報反推 GHI（氣溫用示範曲線）> 全部示範 ----
    w = _prepare_weather(df_weather) if df_weather is not None else None
    ghi_s = pv_ghi_series(pv_df, interval_ending)
    others = [c for c in d.columns if c != load_col]
    t_in = _find_col(others, TEMP_KEYS)
    g_in = _find_col(others, GHI_KEYS)
    temp = ghi = None

    if t_in is not None and g_in is not None:
        temp = pd.to_numeric(d[t_in], errors="coerce").to_numpy(float)
        ghi = pd.to_numeric(d[g_in], errors="coerce").to_numpy(float)
        notes.append(("info", f"天氣來源：大樓 CSV 內的欄位（{t_in}、{g_in}）"))
    elif w is not None and idx is not None:
        hw = w.reindex(idx)
        if hw.isna().any().any():
            if not demo:
                raise ValueError(
                    f"天氣檔在歷史視窗（{idx[0]} ~ {idx[-1]}）內缺 "
                    f"{int(hw.isna().any(axis=1).sum())} 筆，請確認天氣檔涵蓋這段時間。")
            notes.append(("info", "天氣檔沒有涵蓋用電資料的視窗，改用下一種天氣來源。"))
        else:
            temp, ghi = hw["temperature"].to_numpy(float), hw["ghi"].to_numpy(float)
            notes.append(("info", "天氣來源：天氣檔（已依時間與用電資料對齊）"))

    if temp is None and ghi_s is not None and idx is not None:
        g, how = None, ""
        exact = ghi_s.reindex(idx).to_numpy(float)
        if np.isfinite(exact).all():
            g, how = exact, "依時間與用電資料對齊"
        elif demo:
            for ref in (target_day - pd.Timedelta(days=1), target_day):
                g = _ghi_by_time_of_day(ghi_s, idx, ref, interval_ending)
                if g is not None:
                    how = f"套用 {ref.date()} 的 PV 預報 GHI 曲線（依時刻）"
                    break
        if g is not None:
            ghi, temp = g, _demo_temp(L)
            notes.append(("warning",
                          f"⚠️ 目前沒有氣溫資料：GHI 由 PV 預報檔反推（{how}），"
                          "氣溫暫用示範曲線（約 22~28 °C）。有氣溫資料時請上傳天氣 CSV。"))

    if temp is None:
        temp = _demo_temp(L)
        ghi = np.maximum(0, 800 * np.sin(np.linspace(0, np.pi, L)))
        notes.append(("warning", "⚠️ 沒有提供天氣資料，目前使用的是「示範用假天氣」，"
                                 "預測結果不能當作含天氣模型的真實表現。"
                                 "請上傳天氣 CSV 或把天氣檔放進 repo。"))

    # ---- 4. 預測超過一個區塊（例如 2 天）時，需要未來天氣 ----
    future = None
    n_blocks = int(np.ceil(forecast_days * STEPS_PER_DAY / B))
    need = (n_blocks - 1) * B
    if need > 0:
        fidx = pd.date_range(latest_ts + pd.Timedelta(minutes=INTERVAL_MIN),
                             periods=need, freq=f"{INTERVAL_MIN}min")
        if w is not None:
            fw = w.reindex(fidx)
            if fw.notna().all().all():
                future = fw[["temperature", "ghi"]].to_numpy(float)
                notes.append(("info", "第 2 天起使用天氣檔裡的未來天氣。"))
        if future is None and ghi_s is not None:
            fg = ghi_s.reindex(fidx).to_numpy(float)
            if np.isfinite(fg).all():
                future = np.column_stack([_demo_temp(need), fg])
                notes.append(("info", "第 2 天起的 GHI 取自 PV 預報檔反推，氣溫使用示範曲線。"))
        if future is None:
            notes.append(("info", "沒有涵蓋未來日期的天氣，第 2 天起的天氣以歷史最後一天循環代替。"))

    return {
        "matrix": np.column_stack([load, temp, ghi]),
        "timestamps": idx,
        "future_weather": future,
        "demo": demo,
        "notes": notes,
    }


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
# 歷史用電統計（校園電度電費 Excel）
# =========================================================
CAMPUS_SHEET = "107-113各月度數費用合併"
CAMPUS_YEARS = {2018: "107年", 2019: "108年", 2020: "109年", 2021: "110年",
                2022: "111年", 2023: "112年", 2024: "113年"}
CAMPUS_LOCATIONS = ["不分區 (全校總用電)", "行政區", "商學", "校友", "男", "女", "中科", "文創"]


def _campus_year_row(df_excel, target_mco, location):
    """取出某一年、某地點的 12 個月用電度數（找不到回傳 None）。"""
    if location == "不分區 (全校總用電)":
        # 例如 107年 → 找「107度數」區塊裡的「總和」列。
        # 「107費用」區塊也有「總和」列，但那是電費，不能拿來用。
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
                        vals[vals == 0] = np.nan   # 尚未發生的月份在 Excel 裡是 0
                        return vals
        return None

    block = None
    for _, row in df_excel.iterrows():
        val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if target_mco in val and "度數" not in val and "費用" not in val:
            block = target_mco
            continue
        elif any(y in val for y in CAMPUS_YEARS.values()) and val != target_mco:
            block = None
        if block == target_mco and val == location:
            return row.iloc[1:13].values
    return None


def render_history_tab():
    st.header("📈 歷史用電統計與動態趨勢分析")

    with st.expander("📌 點此查看【歷史統計檔案】的規格與欄位說明", expanded=False):
        st.markdown("""
        * **檔案格式**：Excel 試算表（`.xlsx` 或 `.xls`）。
        * **工作表名稱（Sheet）**：需包含各月份用電度數彙整表（例如 `107-113各月度數費用合併`）。
        * **資料排列**：每年有「度數」與「費用」兩個區塊，區塊內有 行政區／商學／校友／男／女／中科／文創／總和 各一列，12 個月各一欄。
        """)

    file = st.file_uploader("請上傳校園電度電費 Excel 統計檔案", type=["xlsx", "xls"], key="hist_excel")
    if file is None:
        st.info("💡 請先上傳您的電度電費 Excel 統計檔案，即可檢視月統計與趨勢圖。")
        return

    try:
        xl = pd.ExcelFile(file)
        default = CAMPUS_SHEET if CAMPUS_SHEET in xl.sheet_names else xl.sheet_names[0]
        sheet = st.selectbox("請選擇資料工作表 (Sheet)", xl.sheet_names,
                             index=xl.sheet_names.index(default), key="hist_sheet")
        df_excel = xl.parse(sheet, header=None)
        st.success(f"✅ 成功讀取檔案：{file.name} (工作表: {sheet})")

        c1, c2 = st.columns(2)
        year = c1.selectbox("選擇分析西元年份", list(CAMPUS_YEARS), index=len(CAMPUS_YEARS) - 1,
                            key="hist_year")
        loc = c2.selectbox("選擇分析地點 / 區域 / 大樓", CAMPUS_LOCATIONS, key="hist_loc")
        mco = CAMPUS_YEARS[year]
        row = _campus_year_row(df_excel, mco, loc)
    except Exception as e:
        st.error(f"❌ 解析上傳的 Excel 檔案時發生錯誤: {e}")
        return

    if row is None:
        st.error(f"❌ 在上傳的檔案中找不到 {year} 年 ({mco}) 對應的【{loc}】資料。")
        return

    months = pd.date_range(start=f"{year}-01-01", periods=12, freq="MS")
    s = pd.Series(pd.to_numeric(pd.Series(row), errors="coerce").to_numpy(float), index=months).dropna()
    if s.empty:
        st.warning(f"⚠️ {year} 年的【{loc}】目前沒有有效的月用電數值。")
        return

    total, mean = s.sum(), s.mean()
    peak_m, low_m = s.idxmax(), s.idxmin()

    st.markdown("---")
    st.subheader(f"📊 {year} 年 ({mco}) - 【{loc}】 用電統計摘要")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("區間總用電量", f"{total:,.2f} kWh")
    m2.metric("每月平均用電", f"{mean:,.2f} kWh")
    m3.metric("用電最高月份", f"{peak_m:%Y-%m}", f"{s.max():,.2f} kWh")
    m4.metric("用電最低月份", f"{low_m:%Y-%m}", f"{s.min():,.2f} kWh")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=f"{loc} Load",
                             line=dict(color="forestgreen", width=3), marker=dict(size=9)))
    fig.add_annotation(x=peak_m, y=s.max(), text=f"Peak: {s.max():,.0f} kWh", showarrow=True,
                       arrowhead=2, ax=0, ay=-40, font=dict(color="crimson", size=13))
    fig.update_layout(
        title=f"{year} {loc} Monthly Electricity Load Analysis",
        xaxis=dict(title=f"Month ({year})", tickmode="array", tickvals=list(s.index),
                   ticktext=[m.strftime("%b") for m in s.index]),
        yaxis_title="Power Consumption (kWh / Month)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


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
params_box = st.sidebar.container()   # 先佔位，等讀完 PV 檔知道可選日期範圍後再填入

st.sidebar.markdown("---")
st.sidebar.subheader("📂 檔案上傳與來源")
uploaded_building = st.sidebar.file_uploader("上傳歷史大樓用電 CSV", type=["csv"])
uploaded_weather = st.sidebar.file_uploader("上傳歷史天氣 CSV（時間、氣溫、GHI）", type=["csv"])
uploaded_pv = st.sidebar.file_uploader("上傳光伏預報 CSV（可多選，每天一個檔案）", type=["csv"],
                                       accept_multiple_files=True)

pv_all, pv_names, pv_errs = get_pv_all(uploaded_pv)
pv_cov = pv_coverage(pv_all)
for _e in pv_errs:
    st.sidebar.error(f"PV 檔讀取失敗：{_e}")
if pv_cov:
    st.sidebar.caption(
        f"PV 預報：{len(pv_names)} 個檔案，涵蓋 {pv_cov['first']} ~ {pv_cov['last']}"
        f"（共 {pv_cov['n_days']} 天）")
    if pv_cov["missing"]:
        st.sidebar.warning("PV 預報缺少或不完整的日期：" + "、".join(str(d) for d in pv_cov["missing"]))
else:
    st.sidebar.caption("尚未找到光伏預報檔。")

with params_box:
    st.header("⚙️ 預測參數設定")
    if pv_cov:
        target_date = st.date_input("預測目標起始日", value=pv_cov["first"],
                                    min_value=pv_cov["first"], max_value=pv_cov["last"])
    else:
        target_date = st.date_input("預測目標起始日", value=datetime(2026, 10, 19).date())

    if pv_cov is None:
        forecast_days = st.slider("預測天數 (Days)", min_value=1, max_value=7, value=1)
    else:
        _run, _d = 0, target_date
        while _d in pv_cov["available"] and _run < 7:
            _run += 1
            _d += timedelta(days=1)
        if _run >= 2:
            forecast_days = st.slider("預測天數 (Days)", min_value=1, max_value=_run, value=1,
                                      help="最多只能預測到 PV 預報涵蓋的最後一天。")
        else:
            forecast_days = 1
            if _run == 0:
                st.caption("預測天數：1 天（目標日沒有完整的 PV 預報）")
            else:
                st.caption("預測天數：1 天（目標日之後沒有更多 PV 預報）")

    time_mode = st.radio(
        "歷史用電的時間標記",
        ["區間結束 (Interval Ending)，與 PV 檔相同", "區間開始 (Interval Start)"],
        help="PV 檔的 00:15 代表 00:00~00:15 這段的結束時間。若你的用電 CSV 也是這種標記，選第一個；"
             "若 00:00 代表這段的開始，選第二個。選錯會讓負載與 PV 差 15 分鐘。",
    )
    interval_ending = time_mode.startswith("區間結束")
    pv_kwp = st.number_input(
        "PV 裝置容量 (kWp)；0 = 直接用 PV 檔內的示範值", min_value=0.0, value=0.0, step=1.0,
        help="PV 檔的示範值只有約 1 kW 的尖峰，輸入實際容量後，會用「每 kWp 出力 × 容量」重新計算。",
    )
    allow_demo = st.checkbox(
        "用電資料沒有涵蓋目標日時，使用示範模式", value=True,
        help="目前用電 CSV 的日期和 PV 預報日期不同。勾選後，會拿用電 CSV 的最後一天當模型輸入，"
             "預測結果標示為目標日。取消勾選則會直接報錯。",
    )

tab_hist, tab1, tab2, tab3, tab4 = st.tabs([
    "📈 歷史用電統計與趨勢分析",
    "🚀 24小時預測與淨負載圖表",
    "📊 批次數據預覽與下載",
    "🧪 系統防禦與稽核統計",
    "📖 說明與規格說明",
])

# =========================================================
# 分頁 0：歷史用電統計
# =========================================================
with tab_hist:
    render_history_tab()

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
                midnight = pd.Timestamp(datetime.combine(target_date, datetime.min.time()))
                # 區間結束：目標日 00:00 這筆屬於前一天最後一段，是最後一筆歷史資料；
                # 預測時間 = 00:15 ~ 隔天 00:00，正好對上 PV 檔的 Step15 = 1~96。
                latest_ts = midnight if interval_ending else midnight - pd.Timedelta(minutes=INTERVAL_MIN)
                pv_offset = 0 if interval_ending else -INTERVAL_MIN
                inp = build_model_inputs(
                    df_build, df_weather, latest_ts, forecast_days,
                    target_day=midnight, pv_df=pv_all,
                    interval_ending=interval_ending, allow_demo=allow_demo,
                )
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
                    if pv_all is None or pv_cov is None:
                        st.session_state["warns"] = ["找不到光伏預報檔，僅顯示總負載預測。"]
                    else:
                        needed = [target_date + timedelta(days=i) for i in range(int(forecast_days))]
                        miss = [d for d in needed if d not in pv_cov["available"]]
                        if miss:
                            raise ValueError(
                                f"PV 預報缺少 {'、'.join(str(d) for d in miss)} 的完整資料"
                                f"（目前涵蓋 {pv_cov['first']} ~ {pv_cov['last']}）。"
                                "請縮短預測天數、改選目標日，或上傳補齊的 PV 檔。")
                        net_df, warns = compute_net_load(
                            forecast_result=forecast_res,
                            pv_forecast=pv_all,
                            pv_col="P_PV_kW_DEMO" if "P_PV_kW_DEMO" in pv_all.columns else "P_PV_kW",
                            time_col="IntervalEnding",
                            pv_time_offset_min=pv_offset,
                            pv_capacity_kwp=(float(pv_kwp) if pv_kwp > 0 else None),
                        )
                        st.session_state["net_df"] = net_df
                        st.session_state["warns"] = (
                            [f"PV 預報來源：{len(pv_names)} 個檔案，涵蓋 {pv_cov['first']} ~ {pv_cov['last']}"]
                            + warns)
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
    - **光伏預報檔**：可以放很多個（每天一個），網站會自動合併、去除重複日期，並只允許選擇有完整 96 筆的日期。
    - **示範模式**：用電 CSV 的日期沒有涵蓋目標日時，會拿用電 CSV 的最後一天當輸入，結果只能用來展示流程，不能當作真實預測績效。
    - **GHI 反推**：沒有天氣檔時，GHI 由 PV 預報檔的「每 kWp 出力」反推，氣溫則用示範曲線。
    """)