# core/logging.py
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from .config import get_settings

settings = get_settings()


def setup_logging():
    """
    简单的 logging 配置。会输出到控制台，并在 settings.LOG_FILE 指定时写入滚动文件。
    如果你需要 JSON 日志或更复杂结构化日志，可替换 formatter。
    """
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # 清理预先存在的 handlers（避免重复）
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # 可选：写日志文件（滚动）
    if settings.LOG_FILE:
        fh = RotatingFileHandler(settings.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # 降低某些库的日志噪音（按需）
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
