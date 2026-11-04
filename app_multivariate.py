import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from infer_multivariate import MultivariateEMSPredictor, compute_net_load

st.set_page_config(
    page_title="微電網 EMS 負載預測與調度系統 (多變數氣象整合版)",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_predictor():
    return MultivariateEMSPredictor(
        model_path='lstm_model_multivariate.h5',
        scaler_path='data_scaler_multivariate.pkl'
    )

predictor = load_predictor()

st.title("⚡ 微電網 EMS 智慧負載預測與淨負載對接平台")
st.markdown("---")

if not predictor.is_ready:
    st.error(f"❌ 模型載入失敗: {predictor.init_error}")
else:
    st.success("✅ 多變數 3D LSTM 模型（歷史負載、氣溫、GHI）已成功載入並就緒！")

# Sidebar
st.sidebar.header("⚙️ 預測參數設定")
forecast_days = st.sidebar.slider("預測天數 (Days)", min_value=1, max_value=7, value=1)
target_date = st.sidebar.date_input("預測目標起始日", value=datetime(2026, 10, 19).date())

st.sidebar.markdown("---")
st.sidebar.subheader("📂 檔案上傳與來源")
uploaded_building = st.sidebar.file_uploader("上傳歷史大樓用電 CSV", type=["csv"])
uploaded_pv = st.sidebar.file_uploader("上傳光伏預報 CSV", type=["csv"])

# Main Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 24小時預測與淨負載圖表", "📊 批次數據預覽與下載", "🧪 系統防禦與稽核統計", "📖 說明與規格說明"])

with tab1:
    st.subheader("🌐 總負載 (Gross Load) 與淨負載 (Net Load) 即時預測")
    
    if st.button("🚀 執行多變數智慧預測與對接運算", type="primary"):
        with st.spinner("正在進行 3 特徵 LSTM 模型推論與淨負載計算..."):
            try:
                # 1. Load building data
                if uploaded_building is not None:
                    df_build = pd.read_csv(uploaded_building)
                else:
                    df_build = pd.read_csv("Report251014_m_buildings.csv")
                
                # Check column
                load_col = '人言大樓kw' if '人言大樓kw' in df_build.columns else df_build.columns[1]
                hist_load = df_build[load_col].tail(96).to_numpy(dtype=float)
                
                # Synthetic weather features for demo/testing if not provided
                hist_temp = 25.0 + 3.0 * np.sin(np.linspace(0, 2 * np.pi, 96))
                hist_ghi = np.maximum(0, 800 * np.sin(np.linspace(0, np.pi, 96)))
                
                history_matrix = np.column_stack([hist_load, hist_temp, hist_ghi])
                latest_ts = datetime.combine(target_date, datetime.min.time()) - timedelta(minutes=15)
                
                # Run forecast
                forecast_res = predictor.forecast_load_multivariate(
                    recent_history_matrix=history_matrix,
                    latest_data_timestamp=latest_ts,
                    forecast_days=forecast_days
                )
                
                # 2. PV Forecast
                if uploaded_pv is not None:
                    pv_file = uploaded_pv
                else:
                    pv_file = "PV_Forecast_15min_20261019.csv"
                
                if os.path.exists(pv_file) or uploaded_pv is not None:
                    net_df, warns = compute_net_load(
                        forecast_result=forecast_res,
                        pv_forecast=pv_file,
                        pv_col="P_PV_kW_DEMO" if "P_PV_kW_DEMO" in pd.read_csv(pv_file).columns else "P_PV_kW",
                        time_col="IntervalEnding" if "IntervalEnding" in pd.read_csv(pv_file).columns else "timestamp"
                    )
                    
                    st.session_state['net_df'] = net_df
                    st.session_state['warns'] = warns
                    st.session_state['forecast_res'] = forecast_res
                    
                    st.success("🎉 預測與淨負載計算完成！")
                else:
                    st.warning("⚠️ 找不到光伏預報檔，僅顯示總負載預測。")
                    st.session_state['forecast_res'] = forecast_res
                    st.session_state['net_df'] = None
                    
            except Exception as e:
                st.error(f"❌ 運算過程發生錯誤: {str(e)}")

    # Display results if available
    if 'net_df' in st.session_state and st.session_state['net_df'] is not None:
        ndf = st.session_state['net_df']
        warns = st.session_state.get('warns', [])
        
        for w in warns:
            st.warning(f"⚠️ {w}")
            
        # Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ndf.index, y=ndf['gross_load_kw'], name="Gross Load (預測總負載)", line=dict(color='crimson', width=2.5)))
        fig.add_trace(go.Scatter(x=ndf.index, y=ndf['pv_kw'], name="PV Forecast (光伏預報)", line=dict(color='orange', width=2.5)))
        fig.add_trace(go.Scatter(x=ndf.index, y=ndf['net_load_kw'], name="Net Load (淨負載 = 總負載 - PV)", line=dict(color='dodgerblue', width=2.5, dash='dash')))
        
        fig.update_layout(
            title="微電網 15 分鐘級電力功率流向預測 (多變數氣象整合版)",
            xaxis_title="時間戳記 (Interval Ending)",
            yaxis_title="功率 (kW)",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics summary
        col1, col2, col3 = st.columns(3)
        col1.metric("平均總負載", f"{ndf['gross_load_kw'].mean():.2f} kW")
        col2.metric("平均光伏發電", f"{ndf['pv_kw'].mean():.2f} kW")
        col3.metric("平均淨負載", f"{ndf['net_load_kw'].mean():.2f} kW")

with tab2:
    st.subheader("📊 預測輸出 CSV 檔案檢視與下載")
    if 'net_df' in st.session_state and st.session_state['net_df'] is not None:
        ndf = st.session_state['net_df']
        st.dataframe(ndf, use_container_width=True)
        
        csv_data = ndf.reset_index().to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下載 EMS 標準對接輸出檔 (CSV)",
            data=csv_data,
            file_name=f"EMS_Output_Multivariate_{target_date}.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 請先至「24小時預測與淨負載圖表」頁籤點擊執行預測按鈕以產生數據。")

with tab3:
    st.subheader("🧪 系統防禦模組與稽核統計")
    stats = predictor.get_system_audit_report()
    st.json(stats)

with tab4:
    st.subheader("📖 多變數模型說明與規格")
    st.markdown("""
    - **模型架構**：3 特徵輸入 (`load_kw`, `temperature`, `ghi`)，透過雙層 LSTM 與 Dense 輸出未來 96 步（24小時）預測。
    - **防禦機制**：包含 NaN 檢查、負值攔截、氣溫與 GHI 合理性檢查、以及自動備援機制。
    - **淨負載定義**：$	ext{Net Load} = 	ext{Gross Load} - 	ext{PV}$，可為負值（支援儲能充電與反送電）。
    """)
