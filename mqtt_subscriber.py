# mqtt_subscriber.py
import json
import os
import uuid
from pathlib import Path
from datetime import datetime
import pandas as pd
import paho.mqtt.client as mqtt
from mqtt_contract import MQTTContractValidator

class MQTTSubscriber:
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
        
        self.stale_timeout = self.config.get("stale_timeout_seconds", 120.0)
        self.validation_mode = self.config.get("validation_mode", "live")
        if self.validation_mode not in {"live", "replay"}:
            raise ValueError(f"Invalid validation_mode: '{self.validation_mode}'. Must be strictly 'live' or 'replay'.")

        self.last_accepted_prediction_timestamp = None
        self.last_received_signature = None

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[Subscriber] ✅ 成功連線至 Broker ({self.host}) [Mode: {self.validation_mode.upper()}]")
            client.subscribe(self.topic, qos=self.qos)
            print(f"[Subscriber] 📡 已訂閱 Topic: {self.topic}")

    def on_message(self, client, userdata, msg):
        print(f"\n[Subscriber] 📥 收到來自 Topic [{msg.topic}] 的訊息")
        
        try:
            payload_str = msg.payload.decode("utf-8")
            payload_dict = json.loads(payload_str)
        except Exception:
            print("[FORECAST REJECT] reason=malformed JSON bytes")
            return
        
        validator = MQTTContractValidator()
        is_valid, err_msg = validator.validate_payload(payload_dict)
        
        if not is_valid:
            print("[CONTRACT REJECT] reason=" + err_msg)
            return
            
        ai_valid = payload_dict.get("ai_valid")
        if ai_valid == 0:
            print("[AI INVALID / FALLBACK]")
            print(f"reason=ai_valid is 0, error={payload_dict.get('error')}")
            return

        # ==========================================
        # Gate F6 & F7: Stale / Ordering / Duplicate & Mode Check
        # ==========================================
        now_dt = datetime.now().astimezone()

        signature_dict = {
            "schema_version": payload_dict.get("schema_version"),
            "forecast_load_kw": payload_dict.get("forecast_load_kw"),
            "latest_data_timestamp": payload_dict.get("latest_data_timestamp"),
            "prediction_timestamp": payload_dict.get("prediction_timestamp"),
            "prediction_horizon_min": payload_dict.get("prediction_horizon_min"),
            "model_version": payload_dict.get("model_version"),
            "ai_valid": payload_dict.get("ai_valid")
        }
        current_signature = json.dumps(signature_dict, sort_keys=True)

        if current_signature == self.last_received_signature:
            print("[DUPLICATE RECORDED] ignored.")
            return

        try:
            gen_at_dt = pd.to_datetime(payload_dict.get("generated_at"))
            latest_dt = pd.to_datetime(payload_dict.get("latest_data_timestamp"))
            pred_dt = pd.to_datetime(payload_dict.get("prediction_timestamp"))
        except Exception as e:
            print(f"[CONTRACT REJECT] timestamp parsing failed: {str(e)}")
            return

        # 若為 Live Mode，才進行 Wall-clock Freshness 與 Expiration 檢查
        if self.validation_mode == "live":
            age_gen = (now_dt - gen_at_dt).total_seconds()
            if age_gen > self.stale_timeout:
                print(f"[STALE REJECT] generated_at too old ({age_gen:.1f}s > {self.stale_timeout}s)")
                return

            if pred_dt <= now_dt:
                print(f"[EXPIRED REJECT] prediction_timestamp is in the past.")
                return
        else:
            print("[REPLAY MODE ACTIVE] Bypassing wall-clock freshness checks.")

        # Ordering 檢查
        if self.last_accepted_prediction_timestamp is not None:
            if pred_dt <= self.last_accepted_prediction_timestamp:
                print("[OUT_OF_ORDER REJECT] prediction_timestamp older than last accepted.")
                return

        self.last_accepted_prediction_timestamp = pred_dt
        self.last_received_signature = current_signature

        forecast_array = payload_dict.get("forecast_load_kw")
        print("[FRESH / REPLAY -> FORECAST ACCEPT]")
        print(f"   👉 24小時預測成功！共接收到 {len(forecast_array)} 個時間點的負載數值 (kW)")
        print(f"   👉 前三筆預測範例: {forecast_array[:3]} ... (略)")
        print(f"   👉 預測起始時間戳記 = {payload_dict.get('prediction_timestamp')}")

    def start_listening(self):
        client_id = f"eams_ai_sub_{uuid.uuid4().hex[:8]}"
        client = mqtt.Client(client_id=client_id)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        
        try:
            client.connect(self.host, self.port, keepalive=60)
            client.loop_forever()
        except KeyboardInterrupt:
            print("\n[Subscriber] 🛑 已手動停止。")
        except Exception as e:
            print(f"[Subscriber] ❌ 連線異常: {str(e)}")

if __name__ == "__main__":
    subscriber = MQTTSubscriber()
    subscriber.start_listening()