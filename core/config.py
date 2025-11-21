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
