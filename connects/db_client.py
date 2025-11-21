from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pandas as pd
from configs.db_config import tenant_db_params


class DBClient:
    def __init__(self, db_config: dict = tenant_db_params):
        self.password_encoded = quote_plus(db_config["password"])
        self.user = db_config["user"]
        self.host = db_config["host"]
        self.port = db_config["port"]
        self.database = db_config["database"]
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = create_engine(
                f"mysql+pymysql://{self.user}:{self.password_encoded}@{self.host}:{self.port}/{self.database}",
                pool_recycle=3600)

    def read_sql(self, sql: text, params: dict = None) -> pd.DataFrame:
        """
        通用接口：任意 SQL + 参数，返回 DataFrame
        TempService 直接调用这个方法查询表即可。
        """
        self._ensure_engine()
        params = params or {}
        return pd.read_sql(sql, self._engine, params=params)

    def close(self):
        if self._engine is not None:
            try:
                self._engine.dispose()
            finally:
                self._engine = None
