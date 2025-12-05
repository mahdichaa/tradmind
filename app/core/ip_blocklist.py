# app/core/ip_blocklist.py
import json, os, tempfile, threading
from ipaddress import ip_address, ip_network
from typing import List

class IPBlocklistStore:
    """
    JSON-file backed store for blocked IPs/CIDRs.
    File format: {"ips": ["1.2.3.4", "10.0.0.0/8", ...]}
    """
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._nets = []  # cached parsed networks

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._atomic_write({"ips": []})
        self._load()

    def _atomic_write(self, data: dict):
        fd, tmp = tempfile.mkstemp(prefix="blocked_ips_", dir=os.path.dirname(self.path))
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

    def _load(self):
        with self._lock:
            with open(self.path, "r") as f:
                obj = json.load(f)
            ips: List[str] = obj.get("ips", [])
            self._nets = []
            for item in ips:
                # single IP becomes /32 or /128 automatically
                if "/" in item:
                    self._nets.append(ip_network(item, strict=False))
                else:
                    self._nets.append(ip_network(str(ip_address(item))))
    
    def _save(self, items: List[str]):
        with self._lock:
            self._atomic_write({"ips": items})
            self._load()

    def list(self) -> List[str]:
        with self._lock:
            # read raw strings from file
            with open(self.path, "r") as f:
                return json.load(f).get("ips", [])

    def add(self, item: str):
        # validate first
        _ = ip_network(item, strict=False) if "/" in item else ip_network(str(ip_address(item)))
        with self._lock:
            current = set(self.list())
            current.add(item)
            self._save(sorted(current))

    def remove(self, item: str):
        with self._lock:
            current = set(self.list())
            if item in current:
                current.remove(item)
                self._save(sorted(current))

    def is_blocked(self, client_ip: str) -> bool:
        ip = ip_address(client_ip)
        with self._lock:
            return any(ip in net for net in self._nets)
