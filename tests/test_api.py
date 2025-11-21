import unittest
from fastapi.testclient import TestClient


from main import app


class APITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/")
        self.assertIn(r.status_code, (200, 204))

    def test_predict_sample(self):
        payload = {
            "vehicle_code": "DT2579-FBG-1000133",
            "step_id": "2"
        }
        r = self.client.post("/temp/pack-temp-corr", json=payload)  # 替换为真实接口路径
        self.assertIn(r.status_code, (200, 201))
        try:
            body = r.json()
            self.assertIsInstance(body, dict)
        except ValueError:
            self.fail("返回不是 JSON")
