from kafka import KafkaProducer
import json
from typing import Iterable
# from configs.kafka_config import kafka_params


class KafkaClient:
    def __init__(self, bootstrap_servers: list, topic: str):
        self.bootstrap_servers = bootstrap_servers.split(",") if isinstance(bootstrap_servers, str) else bootstrap_servers
        self.topic = topic
        self._producer = None

    def _ensure_producer(self):
        if self._producer is None:
            self._producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers)

    def send(self, payload: Iterable[dict]):
        self._ensure_producer()
        for item in payload:
            self._producer.send(self.topic, value=json.dumps(item, ensure_ascii=False).encode("utf-8"))
            self._producer.flush()

    def close(self):
        if self._producer is not None:
            self._producer.close()
            self._producer = None
