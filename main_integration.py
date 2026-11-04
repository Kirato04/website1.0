# =========================================================
# 微電網 EMS 系統 - 整合主程式 (main_integration.py)
# =========================================================

import numpy as np
from infer02 import EMSLoadPredictor  # 直接引入你原本寫好的防禦與預測類別

class IntegratedEMSManager:
    def __init__(self):
        # 1. 初始化你原本的負載預測器
        self.load_predictor = EMSLoadPredictor()
        print("✅ 負載預測模組已成功載入！")

    def process_with_weather(self, recent_history_kw, latest_timestamp, weather_forecast_data=None):
        """
        整合氣象資料與負載預測的主控流程
        """
        print("\n--------------------------------------------------")
        print("🚀 正在執行系統整合推論流程...")

        # ==================== 步驟 A：檢查並防禦天氣資料（預留擴充） ====================
        weather_valid = True
        if weather_forecast_data is None:
            print("⚠️ 警告：未收到氣象夥伴的天氣資料，啟動自動降級（Graceful Degradation）")
            weather_valid = False
        else:
            # 這裡未來可以寫天氣資料的防禦檢查（例如檢查溫度是否合理、有無 NaN 等）
            print("🌤️ 成功接收天氣預測資料，準備進行多變量特徵融合...")

        # ==================== 步驟 B：呼叫你原本強大的負載預測模組 ====================
        # 呼叫你已經寫好、具備五大防禦與 ai_valid 機制的函數
        result = self.load_predictor.forecast_load_kw(recent_history_kw, latest_timestamp)

        # ==================== 步驟 C：封裝雙模組的整合結果 ====================
        integrated_output = {
            "ai_valid": result.get("ai_valid"),
            "forecast_load_kw": result.get("forecast_load_kw"),
            "prediction_timestamp": result.get("prediction_timestamp"),
            "error_reason": result.get("error"),
            "fallback_strategy": result.get("fallback_strategy"),
            "module_status": {
                "load_prediction": "active" if result.get("ai_valid") == 1 else "fallback_triggered",
                "weather_integration": "active" if weather_valid else "degraded_pure_load"
            }
        }

        return integrated_output

# ==================== 測試執行區塊 ====================
if __name__ == "__main__":
    # 建立整合管理器實例
    manager = IntegratedEMSManager()

    # 準備一筆正常的測試資料
    normal_history = np.linspace(500, 800, 200).tolist()
    timestamp = "2026-06-07 12:00:00"

    # 模擬狀況 1：氣象夥伴還沒好（無天氣資料，系統自動降級以純負載運作）
    print("\n--- [情境 1] 天氣資料尚未就緒，測試自動降級 ---")
    res_no_weather = manager.process_with_weather(normal_history, timestamp, weather_forecast_data=None)
    print("📋 系統整合輸出結果：")
    print(res_no_weather)

    # 模擬狀況 2：氣象夥伴未來交給你的天氣資料格式
    mock_weather = {
        "temperature": [28.5, 29.0, 29.5] # 假設的未來溫度預測
    }
    print("\n--- [情境 2] 天氣資料已對接，正常多變量協同 ---")
    res_with_weather = manager.process_with_weather(normal_history, timestamp, weather_forecast_data=mock_weather)
    print("📋 系統整合輸出結果：")
    print(res_with_weather)