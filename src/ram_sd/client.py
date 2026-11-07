"""Recorded OpenAI requests; replay never creates a network client."""

import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from .io import digest, read_json, write_json


class CacheMiss(RuntimeError):
    pass


def load_env_file(path):
    """Explicit opt-in to a project env file. Values are never printed or serialized."""
    allowed = {"OPENAI_API_KEY", "RAM_SD_API_KEY", "RAM_SD_EMBEDDING_API_KEY", "RAM_SD_BASE_URL", "RAM_SD_EMBEDDING_BASE_URL"}
    assignments = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in allowed:
            raise ValueError(f"Unsupported env assignment on line {number}; use the keys in .env.example")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not value:
            raise ValueError(f"Empty env value for {name}")
        assignments[name] = value
    os.environ.update(assignments)


def endpoint(value):
    parts = urlsplit(value)
    if parts.scheme not in ("https", "http") or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("API base URL must have no embedded credentials, query, or fragment")
    return value.rstrip("/")


def safe_error(error):
    text = str(error)
    for name in ("OPENAI_API_KEY", "RAM_SD_API_KEY", "RAM_SD_EMBEDDING_API_KEY"):
        value = os.getenv(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)[:1500]


class RecordedClient:
    def __init__(self, cache_dir, mode="replay", transport=None, chat_endpoint=None, embedding_endpoint=None):
        if mode not in ("live", "replay"):
            raise ValueError("mode must be live or replay")
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.transport = transport
        self.chat_endpoint = endpoint(chat_endpoint or os.getenv("RAM_SD_BASE_URL", "https://api.openai.com/v1"))
        self.embedding_endpoint = endpoint(embedding_endpoint or os.getenv("RAM_SD_EMBEDDING_BASE_URL", self.chat_endpoint))
        self.events = []
        self._sdk_clients = {}

    def _sdk(self, kind):
        if kind not in self._sdk_clients:
            from openai import OpenAI
            key = ((os.getenv("RAM_SD_EMBEDDING_API_KEY") if kind == "embedding" else None)
                   or os.getenv("RAM_SD_API_KEY") or os.getenv("OPENAI_API_KEY"))
            if not key:
                raise ValueError("Set OPENAI_API_KEY or RAM_SD_API_KEY for live requests")
            self._sdk_clients[kind] = OpenAI(
                api_key=key, base_url=self.embedding_endpoint if kind == "embedding" else self.chat_endpoint,
                timeout=60, max_retries=2,
            )
        return self._sdk_clients[kind]

    def _request(self, kind, payload, scope):
        url = self.embedding_endpoint if kind == "embedding" else self.chat_endpoint
        identity = {"kind": kind, "endpoint": url, "scope": scope, "payload": payload}
        key = digest(identity)
        path = self.cache_dir / kind / f"{key}.json"
        start = time.monotonic()
        if path.exists():
            record = read_json(path)
            if record.get("identity") != identity or record.get("response_sha256") != digest(record.get("response")):
                raise ValueError(f"Corrupt or incompatible response cache: {key}")
            response = record["response"]
            cached = True
        else:
            if self.mode == "replay":
                raise CacheMiss(f"No saved {kind} response for {scope} ({key}); replay performs no API calls")
            try:
                if self.transport:
                    response = self.transport(kind, payload, scope)
                elif kind == "chat":
                    response = self._sdk(kind).chat.completions.create(**payload).model_dump(mode="json")
                else:
                    response = self._sdk(kind).embeddings.create(**payload).model_dump(mode="json")
            except Exception as error:
                raise RuntimeError(safe_error(error)) from None
            write_json(path, {"identity": identity, "response": response, "response_sha256": digest(response)})
            cached = False
        self.events.append({"key": key, "kind": kind, "scope": scope, "cached": cached,
                            "elapsed_seconds": time.monotonic() - start,
                            "model": response.get("model"), "system_fingerprint": response.get("system_fingerprint"),
                            "response_id": response.get("id"), "usage": response.get("usage")})
        return response

    def chat(self, prompt, config, scope):
        payload = {**config.generation(), "messages": [{"role": "user", "content": prompt}]}
        response = self._request("chat", payload, scope)
        choices = response.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Chat response missing, refused, filtered or truncated; no fallback prediction")
        message = choices[0].get("message", {})
        content = message.get("content")
        if message.get("refusal") or not isinstance(content, str) or not content.strip():
            raise ValueError("Empty or refused model response")
        return content.strip()

    def embed(self, texts, config, scope):
        import numpy as np
        payload = {"model": config.embedding_model, "dimensions": config.embedding_dimensions,
                   "input": list(texts), "encoding_format": "float"}
        response = self._request("embedding", payload, scope)
        items = response.get("data", [])
        if len(items) != len(texts) or sorted(item.get("index", -1) for item in items) != list(range(len(texts))):
            raise ValueError("Embedding response has missing or duplicate indexes")
        vectors = np.asarray([item["embedding"] for item in sorted(items, key=lambda x: x["index"])], dtype="float32")
        if vectors.shape != (len(texts), config.embedding_dimensions) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding dimensions or non-finite vectors")
        if (np.linalg.norm(vectors, axis=1) <= 0).any():
            raise ValueError("Zero embedding vector; refusing to build or query an index")
        return vectors

    def close(self):
        for client in self._sdk_clients.values():
            client.close()
