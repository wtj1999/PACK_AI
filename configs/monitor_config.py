from core.config import get_settings

settings = get_settings()
MONITOR = settings.MONITOR
RECIPIENTS = MONITOR.recipients
BAD_REQUEST_RECIPIENTS = MONITOR.bad_request_recipients
LOCAL_RECIPIENTS = MONITOR.local_recipients
