from core.config import get_settings

settings = get_settings()

LOG_FILE = settings.LOG_FILE or "ai_pack_server.log"
LOG_NAME_SERVICE = "ai_pack_service"
LOG_BAD_REQUEST = "bad_request"
LOCAL_LOG = "local_log"

MAIL_CONFIG = settings.MAIL_CONFIG
CREDENTIALS = settings.CREDENTIALS
