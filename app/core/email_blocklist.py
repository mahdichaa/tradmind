# app/core/email_blocklist.py
import json, os, tempfile, threading
from typing import List

class EmailBlocklistStore:
    """
    JSON-file backed store for blocked emails.
    File format: {"emails": ["user@example.com", ...]}
    """
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._atomic_write({"emails": []})

    def _atomic_write(self, data: dict):
        fd, tmp = tempfile.mkstemp(prefix="blocked_emails_", dir=os.path.dirname(self.path))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except FileNotFoundError:
                pass

    def list(self) -> List[str]:
        with self._lock:
            with open(self.path, "r") as f:
                return json.load(f).get("emails", [])

    def _save(self, items: List[str]):
        with self._lock:
            self._atomic_write({"emails": items})

    @staticmethod
    def _validate(email: str):
        if not email or "@" not in email:
            raise ValueError("Invalid email")

    def add(self, email: str):
        self._validate(email)
        with self._lock:
            current = set(self.list())
            current.add(email.lower())
            self._save(sorted(current))

    def remove(self, email: str):
        with self._lock:
            current = set(self.list())
            email = email.lower()
            if email in current:
                current.remove(email)
                self._save(sorted(current))

    def is_blocked(self, email: str) -> bool:
        if not email:
            return False
        email = email.lower()
        with self._lock:
            return email in set(self.list())
