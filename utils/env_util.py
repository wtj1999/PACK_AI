import os
import socket
from enum import Enum

PROD_HOSTNAMES = ('ai-machine-10.34.195.54', 'node2.itcast.cn', 'jzpack03-docker', '10.2.160.69-docker', 'docker')

ENV_TO_DB_ENV_MAP = {
    'prod': 'prod',
    'pre': 'prod',
    'dev': 'real',
    'test': 'test',
}

ENV_TO_WRITE_DB_ENV_MAP = {
    'prod': 'prod',  # NOTE: 仅在生产环境下写生产库，其他环境只能写测试库
    'pre': 'test',
    'dev': 'test',
    'test': 'test',
}


class Env(Enum):
    PROD = 'prod'
    PRE = 'pre'
    DEV = 'dev'
    TEST = 'test'


class EnvUtil(object):
    @staticmethod
    def is_valid_env(env: str) -> bool:
        if EnvUtil.is_prod_env(env) or EnvUtil.is_pre_env(env) or EnvUtil.is_dev_env(env) or EnvUtil.is_test_env(env):
            return True
        else:
            return False

    @staticmethod
    def get_hostname(default: str = '') -> str:
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = None
        return hostname if hostname else default

    @staticmethod
    def map_to_db_env(env: str) -> str:
        return ENV_TO_DB_ENV_MAP[env]

    @staticmethod
    def map_to_write_db_env(env: str) -> str:
        return ENV_TO_WRITE_DB_ENV_MAP[env]

    @staticmethod
    def is_prod_env(env: str) -> bool:
        is_valid_prod_env = (env == Env.PROD.value)
        is_prod_hosts = (EnvUtil.get_hostname() in PROD_HOSTNAMES or EnvUtil.is_on_cloud())
        return is_valid_prod_env and is_prod_hosts

    @staticmethod
    def is_pre_env(env: str) -> bool:
        return env == Env.PRE.value

    @staticmethod
    def is_dev_env(env: str) -> bool:
        return env == Env.DEV.value

    @staticmethod
    def is_test_env(env: str) -> bool:
        return env == Env.TEST.value

    @staticmethod
    def is_on_cloud() -> bool:
        return False

    @staticmethod
    def mkdir_path(dirs: str) -> None:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
