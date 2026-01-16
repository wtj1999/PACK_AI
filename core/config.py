from typing import Dict, Optional
from pydantic import BaseSettings, Field, BaseModel
from pathlib import Path
from functools import lru_cache


BASE_DIR = Path(__file__).resolve().parent.parent

class DBParams(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str

class TenantDBConfig(BaseModel):
    prod: DBParams
    test: DBParams

class KafkaParams(BaseModel):
    bootstrap_servers: list

class TenantKafkaConfig(BaseModel):
    prod: KafkaParams
    test: KafkaParams

class MonitorConfig(BaseModel):
    recipients: Dict[str, str] = {}
    bad_request_recipients: Dict[str, str] = {}
    local_recipients: Dict[str, str] = {}

class Settings(BaseSettings):
    # 基本环境
    APP_ENV: str = Field("test", env="APP_ENV")   # e.g. "prod" or "test" or "dev"
    TENANT: str = Field("jz2_pack", env="TENANT")
    APP_NAME: str = "ai_pack"
    DEBUG: bool = True

    # Redis
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    # Model & storage
    MODEL_STORE_DIR: Path = Field(BASE_DIR / "models")

    # Logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field("ai_pack_server.log", env="LOG_FILE")

    PACK_CONFIG = {
        "CELLS_PER_PHYSICAL_PACK": 102,
    }

    MODEL_CONFIG = {
        "model_name": 'Catboost',
        "input_feature": ['capacity', 'ocv3', 'ocv4', 'acr3', 'acr4', 'k_value', 'cell_thickness', 'weight']
                              + [f'step_{i}_volt' for i in range(1, 10)],
        "target_name": ['Discharge_Dynamic_Voltage', 'Discharge_Static_Voltage'],
        "out_dim": 1,
        "node_dim": 102,
        "model_store_dir": 'services/result_analysis_service/model_store',
        "DeepSet": {
            "emb_dim": 128
        },
        "Transformer": {
            "model_dim": 128,
            "num_layers": 6,
            "num_heads": 4,
            "use_pack_token": False
        },
        "Catboost": {
            'iterations': 50000,
            'learning_rate': 0.01,
            'depth': 4,
            'loss_function': 'RMSE',
            'early_stopping_rounds': 1000,
            'random_seed': 42,
            'verbose': 100,
        }

    }

    TEST_STEP_CONFIG = {
        "330阶梯充一拖四": {
            "1": "测前电压",
            "2": "0.7C充电末端动态电压",
            "3": "0.7C充电末端静态电压",
            "4": "0.4C充电末端动态电压",
            "5": "0.4C充电末端静态电压",
            "6": "0.2C充电末端动态电压",
            "7": "0.2C充电末端静态电压",
            "8": "0.1C充电末端动态电压",
            "9": "0.1C充电末端静态电压",
            "10": "0.5C放电末端动态电压(电测工步10)",
            "11": "0.5C放电末端静态电压(电测工步11)",
            "12": "0.9C放电末端动态电压",
            "13": "0.9C放电末端静态电压",
            "14": "0.5C放电未端动态电压(电测工步14)",
            "15": "0.5C放电未端静态电压(电测工步15)",
            "16": "0.5C充电末端动态电压",
            "17": "0.5C充电末端静态电压"
        },
        "330阶梯充一拖三": {
            "1": "测前电压",
            "2": "0.7C充电末端动态电压",
            "3": "0.7C充电末端静态电压",
            "4": "0.4C充电末端动态电压",
            "5": "0.4C充电末端静态电压",
            "6": "0.2C充电末端动态电压",
            "7": "0.2C充电末端静态电压",
            "8": "0.1C充电末端动态电压",
            "9": "0.1C充电末端静态电压",
            "10": "0.5C放电末端动态电压(电测工步10)",
            "11": "0.5C放电末端静态电压(电测工步11)",
            "12": "0.9C放电末端动态电压",
            "13": "0.9C放电末端静态电压",
            "14": "0.5C放电未端动态电压(电测工步14)",
            "15": "0.5C放电未端静态电压(电测工步15)",
            "16": "0.5C充电末端动态电压",
            "17": "0.5C充电末端静态电压"
        },
        "330一拖四充放电": {
            "1": "测前电压",
            "2": "0.7C充电末端动态电压",
            "3": "0.7C充电末端静态电压",
            "4": "0.4C充电末端动态电压",
            "5": "0.4C充电末端静态电压",
            "6": "0.2C充电末端动态电压",
            "7": "0.2C充电末端静态电压",
            "8": "0.1C充电末端动态电压",
            "9": "0.1C充电末端静态电压",
            "10": "0.5C放电末端动态电压",
            "11": "0.5C放电末端静态电压",
            "12": "0.5C充电末端动态电压",
            "13": "0.5C充电末端静态电压"
        },
        "330一拖三充放电": {
            "1": "测前电压",
            "2": "0.7C充电末端动态电压",
            "3": "0.7C充电末端静态电压",
            "4": "0.4C充电末端动态电压",
            "5": "0.4C充电末端静态电压",
            "6": "0.2C充电末端动态电压",
            "7": "0.2C充电末端静态电压",
            "8": "0.1C充电末端动态电压",
            "9": "0.1C充电末端静态电压",
            "10": "0.5C放电末端动态电压",
            "11": "0.5C放电末端静态电压",
            "12": "0.5C充电末端动态电压",
            "13": "0.5C充电末端静态电压"
        },
        "0.5C满充满放一拖三": {
            "1": "测前电压",
            "2": "0.5C放电末端动态电压(电测工步2)",
            "3": "0.5C放电末端静态电压(电测工步3)",
            "4": "0.5C充电末端动态电压(电测工步4)",
            "5": "0.5C充电末端静态电压(电测工步5)",
            "6": "0.5C放电末端动态电压(电测工步6)",
            "7": "0.5C放电末端静态电压(电测工步7)",
            "8": "0.5C充电末端动态电压(电测工步8)",
            "9": "0.5C充电末端静态电压(电测工步9)"
        },
        "0.5C满充满放一拖四": {
            "1": "测前电压",
            "2": "0.5C放电末端动态电压(电测工步2)",
            "3": "0.5C放电末端静态电压(电测工步3)",
            "4": "0.5C充电末端动态电压(电测工步4)",
            "5": "0.5C充电末端静态电压(电测工步5)",
            "6": "0.5C放电末端动态电压(电测工步6)",
            "7": "0.5C放电末端静态电压(电测工步7)",
            "8": "0.5C充电末端动态电压(电测工步8)",
            "9": "0.5C充电末端静态电压(电测工步9)"
        }
    }

    TEST_PROCESS_CONFIG = {
        "1拖4": {
            "1": "静置",
            "2": "0.7C恒流充电",
            "3": "静置",
            "4": "0.4C恒流充电",
            "5": "静置",
            "6": "0.2C恒流充电",
            "7": "静置",
            "8": "0.1C恒流充电",
            "9": "静置",
            "10": "0.5C恒流放电(电测工步10)",
            "11": "静置",
            "12": "0.9C恒流放电",
            "13": "静置",
            "14": "0.5C恒流放电(电测工步14)",
            "15": "静置",
            "16": "0.5C恒流充电",
            "17": "静置"
        },
        "330阶梯充一拖三": {
            "1": "静置",
            "2": "0.7C恒流充电",
            "3": "静置",
            "4": "0.4C恒流充电",
            "5": "静置",
            "6": "0.2C恒流充电",
            "7": "静置",
            "8": "0.1C恒流充电",
            "9": "静置",
            "10": "0.5C恒流放电(电测工步10)",
            "11": "静置",
            "12": "0.9C恒流放电",
            "13": "静置",
            "14": "0.5C恒流放电(电测工步14)",
            "15": "静置",
            "16": "0.5C恒流充电",
            "17": "静置"
        },
        "330一拖四充放电": {
            "1": "静置",
            "2": "0.7C恒流充电",
            "3": "静置",
            "4": "0.4C恒流充电",
            "5": "静置",
            "6": "0.2C恒流充电",
            "7": "静置",
            "8": "0.1C恒流充电",
            "9": "静置",
            "10": "0.5C恒流放电",
            "11": "静置",
            "12": "0.5C恒流充电",
            "13": "静置"
        },
        "330一拖三充放电": {
            "1": "静置",
            "2": "0.7C恒流充电",
            "3": "静置",
            "4": "0.4C恒流充电",
            "5": "静置",
            "6": "0.2C恒流充电",
            "7": "静置",
            "8": "0.1C恒流充电",
            "9": "静置",
            "10": "0.5C恒流放电",
            "11": "静置",
            "12": "0.5C恒流充电",
            "13": "静置"
        },
        "0.5C满充满放一拖三": {
            "1": "静置",
            "2": "0.5C恒流放电(电测工步2)",
            "3": "静置",
            "4": "0.5C恒流充电(电测工步4)",
            "5": "静置",
            "6": "0.5C恒流放电(电测工步6)",
            "7": "静置",
            "8": "0.5C恒流充电(电测工步8)",
            "9": "静置"
        },
        "0.5C满充满放一拖四": {
            "1": "静置",
            "2": "0.5C恒流放电(电测工步2)",
            "3": "静置",
            "4": "0.5C恒流充电(电测工步4)",
            "5": "静置",
            "6": "0.5C恒流放电(电测工步6)",
            "7": "静置",
            "8": "0.5C恒流充电(电测工步8)",
            "9": "静置"
        }
    }

    DB_CONFIG: Dict[str, TenantDBConfig] = Field(
        {
            "jz2_pack": {
                "prod": {
                    "host": "10.36.94.17",
                    "port": 19030,
                    "user": "gdmo",
                    "password": "gdmo@123!!",
                    "database": "ai_pack",
                },
                "test": {
                    "host": "10.36.94.17",
                    "port": 19030,
                    "user": "gdmo",
                    "password": "gdmo@123!!",
                    "database": "ai_pack",
                },
            }
        }
    )

    KAFKA_CONFIG: Dict[str, TenantKafkaConfig] = Field(
        {
            "jz2_pack": {
                "prod": {
                    "bootstrap_servers": ["10.0.50.20:9092", "10.0.50.21:9092", "10.0.50.22:9092"],
                },
                "test": {
                    "bootstrap_servers": ["10.0.50.20:9092", "10.0.50.21:9092", "10.0.50.22:9092"],
                },
            }
        }
    )

    MONITOR: MonitorConfig = Field(
        MonitorConfig(
            recipients={"prod": "caoxianfeng@gotion.com.cn", "test": "caoxianfeng@gotion.com.cn"},
            bad_request_recipients={"prod": "caoxianfeng@gotion.com.cn", "test": "caoxianfeng@gotion.com.cn"},
            local_recipients={"all": "caoxianfeng@gotion.com.cn", "per": "caoxianfeng@gotion.com.cn"},
        )
    )

    CREDENTIALS: Dict[str, tuple] = Field(
        {
            "prod": ("caoxianfeng@gotion.com.cn", "Ukky2EEMkz69iyit"),
            "test": ("caoxianfeng@gotion.com.cn", "Ukky2EEMkz69iyit"),
        }
    )

    MAIL_CONFIG: Dict = Field({"mail_host": ("smtp.exmail.qq.com", 25), "timeout": 2})

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"

    def get_tenant_db(self, tenant_key: Optional[str] = None, env: Optional[str] = None) -> DBParams:
        tenant_key = tenant_key or self.TENANT
        env = env or self.APP_ENV
        try:
            entry = self.DB_CONFIG[tenant_key]
            if isinstance(entry, dict):
                entry = TenantDBConfig.parse_obj(entry)
            db_params = getattr(entry, env)
            return db_params.dict()
        except KeyError as e:
            raise KeyError(f"DB config not found for tenant={tenant_key} env={env}: {e}")

    def get_tenant_kafka(self, tenant_key: Optional[str] = None, env: Optional[str] = None) -> KafkaParams:
        tenant_key = tenant_key or self.TENANT
        env = env or self.APP_ENV
        try:
            entry = self.KAFKA_CONFIG[tenant_key]
            if isinstance(entry, dict):
                entry = TenantKafkaConfig.parse_obj(entry)
            kafka_params = getattr(entry, env)
            return kafka_params.dict()
        except KeyError as e:
            raise KeyError(f"Kafka config not found for tenant={tenant_key} env={env}: {e}")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
