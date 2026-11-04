import os
import json
import uuid
from pathlib import Path
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt

from infer import forecast_load_kw
from mqtt_contract import MQTTContractValidator

class MQTTPublisher:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = Path(__file__).resolve().parent
            config_path = base_dir / "mqtt_config.json"
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到 MQTT 設定檔: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.host = self.config.get("broker_host", "broker.emqx.io")
        self.port = self.config.get("broker_port", 1883)
        self.topic = self.config.get("topic", "eams/dev/a83f41c2/ai/load_forecast/v1")
        self.qos = self.config.get("qos", 1)
        self.retain = self.config.get("retain", False)

    def publish_forecast(self, recent_history_kw, latest_data_timestamp):
        print("[Publisher] 1. 呼叫 AI 引擎運算中 (24小時 / 96步長)...")
        raw_result = forecast_load_kw(recent_history_kw, latest_data_timestamp)
        
        latest_ts_str = latest_data_timestamp.isoformat() if isinstance(latest_data_timestamp, datetime) else latest_data_timestamp
        
        # 建立 24 小時預測的 MQTT Payload 結構
        payload = {
            "schema_version": "1.0",
            "forecast_load_kw": raw_result.get("forecast_load_kw"), # 這裡會是包含 96 個元素的陣列
            "latest_data_timestamp": latest_ts_str,
            "prediction_timestamp": raw_result.get("prediction_timestamp"),
            "prediction_horizon_min": raw_result.get("prediction_horizon_min", 1440), # 調整為 1440 分鐘 (24小時)
            "model_version": raw_result.get("model_version"),
            "ai_valid": raw_result.get("ai_valid", 0),
            "generated_at": datetime.now().astimezone().isoformat(),
            "error": raw_result.get("error")
        }

        print("[Publisher] 2. 進行 MQTT 合約安檢...")
        validator = MQTTContractValidator()
        is_valid, err_msg = validator.validate_payload(payload)
        
        if not is_valid:
            print(f"[Publisher] ❌ 安檢失敗，拒絕發布！原因: {err_msg}")
            return False

        json_payload = json.dumps(payload, ensure_ascii=False)
        
        print(f"[Publisher] 3. 準備發布至 Broker ({self.host}:{self.port})")
        print(f"[Publisher] Topic: {self.topic}")
        
        try:
            client_id = f"eams_ai_pub_{uuid.uuid4().hex[:8]}"
            client = mqtt.Client(client_id=client_id)
            client.connect(self.host, self.port, keepalive=60)
            client.loop_start()
            
            msg_info = client.publish(self.topic, json_payload, qos=self.qos, retain=self.retain)
            msg_info.wait_for_publish()
            
            client.loop_stop()
            client.disconnect()
            print("[Publisher] ✅ 24小時預測封包發布成功！")
            return True
            
        except Exception as e:
            print(f"[Publisher] ❌ 發布時發生網路異常: {str(e)}")
            return False

if __name__ == "__main__":
    publisher = MQTTPublisher()
    # 測試時提供 192 筆歷史資料（符合新模型的 lookback 需求）
    dummy_history_kw = np.random.uniform(200.0, 600.0, size=192).tolist()
    current_time = datetime.now().astimezone()
    publisher.publish_forecast(dummy_history_kw, current_time)