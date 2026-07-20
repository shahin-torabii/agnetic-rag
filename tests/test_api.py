import unittest

import conftest  # noqa: F401


_user_counter = 0


def _unique_user():
    global _user_counter
    _user_counter += 1
    return f"testuser{_user_counter}"


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from api.main import app
        cls.client = TestClient(app)

    def _register(self, username=None):
        username = username or _unique_user()
        resp = self.client.post(
            "/auth/register",
            json={"username": username, "password": "pass123"},
        )
        return resp

    def test_register_user(self):
        resp = self._register()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("access_token", resp.json())

    def test_register_duplicate(self):
        name = _unique_user()
        self._register(name)
        resp = self._register(name)
        self.assertEqual(resp.status_code, 400)

    def test_login_valid(self):
        name = _unique_user()
        self._register(name)
        resp = self.client.post(
            "/auth/login",
            json={"username": name, "password": "pass123"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("access_token", resp.json())

    def test_login_invalid(self):
        resp = self.client.post(
            "/auth/login",
            json={"username": "nobody", "password": "wrong"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_chat_no_auth(self):
        resp = self.client.post("/chat", json={"query": "hello"})
        self.assertEqual(resp.status_code, 401)

    @unittest.skip("Requires running model server (HF_LLM.fast_llm)")
    def test_chat_with_auth(self):
        resp = self._register()
        token = resp.json()["access_token"]
        resp = self.client.post(
            "/chat",
            json={"query": "hello", "documents": [], "images": [], "audio": []},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(resp.status_code, 200)

    def test_list_chats(self):
        resp = self._register()
        token = resp.json()["access_token"]
        resp = self.client.get("/chats", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_upload_no_auth(self):
        resp = self.client.post("/upload", files={"file": ("test.txt", b"hello")})
        self.assertEqual(resp.status_code, 401)

    def test_list_documents(self):
        resp = self._register()
        token = resp.json()["access_token"]
        resp = self.client.get("/documents", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)


if __name__ == "__main__":
    unittest.main()
