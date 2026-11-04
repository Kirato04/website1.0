# mqtt_contract.py
import json
import math
from datetime import datetime
import pandas as pd

class MQTTContractValidator:
    def __init__(self):
        self.schema_version = "1.0"

    @staticmethod
    def normalize_timestamp(ts_input):
        if ts_input is None:
            return None
        try:
            if isinstance(ts_input, datetime):
                dt = ts_input
            else:
                if not isinstance(ts_input, str):
                    return None
                dt = pd.to_datetime(ts_input)
            if dt.tzinfo is None:
                return None
            return dt.isoformat()
        except Exception:
            return None

    def validate_payload(self, payload_dict):
        # 2. 修正 JSON 頂層型態防呆：必須是 Python dict / JSON object
        if not isinstance(payload_dict, dict):
            return False, "payload root must be a JSON object"

        # 檢查基本型態與必填欄位
        if payload_dict.get("schema_version") != self.schema_version:
            return False, f"Invalid schema_version, must be '{self.schema_version}'"

        ai_valid = payload_dict.get("ai_valid")
        if ai_valid not in (0, 1):
            return False, "ai_valid must be integer 0 or 1 (strict type check)"

        # 1. 修正 ai_valid=0 Fallback Schema
        if ai_valid == 0:
            if payload_dict.get("forecast_load_kw") is not None:
                return False, "When ai_valid=0, forecast_load_kw MUST be null"
            
            # 允許 prediction_timestamp 為 null 或合法時區 ISO
            pred_ts = payload_dict.get("prediction_timestamp")
            if pred_ts is not None and MQTTContractValidator.normalize_timestamp(pred_ts) is None:
                return False, "When ai_valid=0, prediction_timestamp must be null or a valid tz-aware ISO string"

            err_val = payload_dict.get("error")
            if not err_val or not isinstance(err_val, str) or len(err_val.strip()) == 0:
                return False, "When ai_valid=0, error message MUST be a non-empty string"
            return True, "Valid Fallback Payload"

        # ==========================================
        # ai_valid == 1 的 24 小時 / 96 步長陣列嚴格驗證
        # ==========================================
        val = payload_dict.get("forecast_load_kw")
        
        # 必須是串列 (List)，且長度必須嚴格等於 96 步
        if not isinstance(val, list) or isinstance(val, bool):
            return False, "forecast_load_kw must be a list of 96 numerical values"
        
        if len(val) != 96:
            return False, f"forecast_load_kw length must be exactly 96, got {len(val)}"

        # 逐一檢查 96 個預測值是否皆為合法的非負有限數
        try:
            for item in val:
                if isinstance(item, bool) or isinstance(item, str):
                    return False, "forecast_load_kw elements must be numeric, not bool or str"
                item_float = float(item)
                if math.isnan(item_float) or math.isinf(item_float) or item_float < 0:
                    return False, "all forecast_load_kw elements must be non-negative finite numbers"
        except (ValueError, TypeError):
            return False, "forecast_load_kw elements must be numeric"

        # 預測步長（Horizon）必須是 1440 分鐘（24小時）
        if payload_dict.get("prediction_horizon_min") != 1440:
            return False, "prediction_horizon_min MUST be 1440 for 24-hour forecasting"

        latest_norm = MQTTContractValidator.normalize_timestamp(payload_dict.get("latest_data_timestamp"))
        pred_norm = MQTTContractValidator.normalize_timestamp(payload_dict.get("prediction_timestamp"))
        gen_norm = MQTTContractValidator.normalize_timestamp(payload_dict.get("generated_at"))

        if not latest_norm or not pred_norm or not gen_norm:
            return False, "Timestamps must be ISO 8601 with explicit timezone"

        try:
            latest_dt = pd.to_datetime(latest_norm)
            pred_dt = pd.to_datetime(pred_norm)
            # 24 小時預測的時間差驗證：從 latest 到第一筆 prediction 時間通常應對應一個步長（15分鐘），或是確保時間邏輯合理
            # 這裡我們允許時間差計算正常（對應 24 小時預測開頭）
            diff_mins = (pred_dt - latest_dt).total_seconds() / 60
            if diff_mins < 0:
                return False, "prediction_timestamp cannot be earlier than latest_data_timestamp"
        except Exception as e:
            return False, f"Timestamp computation failed: {str(e)}"

        return True, "Valid 24H Forecast Payload"