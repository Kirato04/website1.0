# =========================================================
# 微電網 EMS 系統 - MQTT 訊息接收與通訊檢查模組 (mqtt_listener.py)
# =========================================================

import json
import paho.mqtt.client as mqtt
from infer02 import forecast_load_kw  # 引入你原本寫好的推論模組

# MQTT 連線設定
MQTT_BROKER = "localhost"  # 或是你的 MQTT Broker IP
MQTT_PORT = 1883
MQTT_TOPIC_INPUT = "microgrid/sensors/history_load"
MQTT_TOPIC_OUTPUT = "microgrid/ai/forecast_result"

def on_connect(client, userdata, flags, rc):
    print(f"🔗 已成功連線至 MQTT Broker，回應碼: {rc}")
    client.subscribe(MQTT_TOPIC_INPUT)
    print(f"📡 已訂閱主題: {MQTT_TOPIC_INPUT}")

def on_message(client, userdata, msg):
    """
    當收到 MQTT 資料時的檢查與處理流程
    """
    print("\n--------------------------------------------------")
    print(lambda_str := f"📥 收到來自主題 [{msg.topic}] 的 MQTT 訊息")
    
    try:
        # ==================== 1. MQTT 封包解析檢查 ====================
        payload = json.loads(msg.payload.decode('utf-8'))
        
        # 檢查是否有必要的欄位
        if "history_kw" not in payload or "timestamp" not in payload:
            print("❌ [MQTT 檢查失敗] 封包格式錯誤：缺少 'history_kw' 或 'timestamp' 欄位")
            return

        raw_history = payload["history_kw"]
        raw_timestamp = payload["timestamp"]

        print("✅ [MQTT 檢查通過] 通訊格式正確，準備交由 AI 推論引擎處理...")

        # ==================== 2. 呼叫你原本的 AI 推論與防禦模組 ====================
        # 這會直接套用你寫好的 5 道防禦機制、ai_valid 判斷與 Fallback 策略
        result = forecast_load_kw(raw_history, raw_timestamp)

        # ==================== 3. 將結果透過 MQTT 發布出去（供上層 EMS 使用） ====================
        print(f"📤 產出預測結果，ai_valid 狀態: {result.get('ai_valid')}")
        # client.publish(MQTT_TOPIC_OUTPUT, json.dumps(result))

    except json.JSONDecodeError:
        print("❌ [MQTT 檢查失敗] JSON 解析錯誤：收到的不是合法的 JSON 格式")
    except Exception as e:
        print(f"❌ [MQTT 處理例外] 發生未預期錯誤: {str(e)}")

def start_mqtt_listener():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("🚀 正在啟動 MQTT 監聽服務...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    # 開始持續監聽迴圈
    client.loop_forever()

if __name__ == "__main__":
    start_mqtt_listener()