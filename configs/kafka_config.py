from core.config import get_settings

settings = get_settings()

kafka_params = settings.get_tenant_kafka()

