from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
import threading
from typing import Any

from triton import knobs
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.runtime.cache import triton_key


_ENV_TRUE = {"1", "true", "yes", "on"}
_DEFAULT_MAX_ENTRIES = 2048
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024
_METADATA_KEYS = (
    "num_warps",
    "shared",
    "tmem_size",
    "global_scratch_size",
    "global_scratch_align",
    "profile_scratch_size",
    "profile_scratch_align",
)


@dataclass(frozen=True)
class WholeLLIRCacheKey:
    key: str
    ttgir_hash: str
    context_hash: str


@dataclass
class WholeLLIRCacheEntry:
    llir: str
    metadata: dict[str, Any]
    ttgir_hash: str
    context_hash: str
    llir_hash: str
    size_bytes: int


class _WholeLLIRCache:

    def __init__(self) -> None:
        self._entries: OrderedDict[str, WholeLLIRCacheEntry] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def get(self, key: WholeLLIRCacheKey) -> WholeLLIRCacheEntry | None:
        with self._lock:
            entry = self._entries.get(key.key)
            if entry is None:
                return None
            self._entries.move_to_end(key.key)
            return entry

    def put(self, key: WholeLLIRCacheKey, llir: str, metadata: dict[str, Any]) -> WholeLLIRCacheEntry:
        entry = WholeLLIRCacheEntry(
            llir=llir,
            metadata={name: metadata.get(name) for name in _METADATA_KEYS},
            ttgir_hash=key.ttgir_hash,
            context_hash=key.context_hash,
            llir_hash=_sha256_text(llir),
            size_bytes=len(llir.encode("utf-8")),
        )
        with self._lock:
            old = self._entries.pop(key.key, None)
            if old is not None:
                self._bytes -= old.size_bytes
            self._entries[key.key] = entry
            self._bytes += entry.size_bytes
            self._evict_locked()
        return entry

    def _evict_locked(self) -> None:
        max_entries = _env_int("HELIONCACHE_TTGIR_LLIR_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES)
        max_bytes = _env_int("HELIONCACHE_TTGIR_LLIR_MAX_BYTES", _DEFAULT_MAX_BYTES)
        if max_entries <= 0 or max_bytes <= 0:
            self._entries.clear()
            self._bytes = 0
            return
        while self._entries and (len(self._entries) > max_entries or self._bytes > max_bytes):
            _, entry = self._entries.popitem(last=False)
            self._bytes -= entry.size_bytes

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._entries), "bytes": self._bytes}


_WHOLE_LLIR_CACHE = _WholeLLIRCache()


def enabled() -> bool:
    return os.environ.get("HELIONCACHE_TTGIR_LLIR_CACHE", "0").lower() in _ENV_TRUE


def cache_level() -> int:
    if not enabled():
        return 0
    value = os.environ.get("HELIONCACHE_TTGIR_LLIR_CACHE_LEVEL")
    if value is None:
        return 1
    try:
        return max(0, int(value))
    except ValueError:
        return 1


def make_whole_llir_key(src, backend_hash: str, options, capability: int, ptx_version: int) -> WholeLLIRCacheKey:
    ttgir_text = _module_str_nodebug(src)
    ttgir_hash = _sha256_text(ttgir_text)
    context = {
        "schema": "helioncache.whole_llir.v1",
        "triton_key": triton_key(),
        "backend_hash": backend_hash,
        "options_hash": options.hash(),
        "capability": capability,
        "target_arch": getattr(options, "arch", None),
        "ptx_version": ptx_version,
        "cache_invalidating_env_vars": sorted(get_cache_invalidating_env_vars().items()),
        "disable_line_info": knobs.compilation.disable_line_info,
        "enable_asan": knobs.compilation.enable_asan,
        "enable_experimental_consan": knobs.compilation.enable_experimental_consan,
    }
    context_json = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    context_hash = _sha256_text(context_json)
    return WholeLLIRCacheKey(
        key=_sha256_text(f"{ttgir_hash}:{context_hash}"),
        ttgir_hash=ttgir_hash,
        context_hash=context_hash,
    )


def lookup_whole_llir(key: WholeLLIRCacheKey) -> WholeLLIRCacheEntry | None:
    return _WHOLE_LLIR_CACHE.get(key)


def store_whole_llir(key: WholeLLIRCacheKey, llir: str, metadata: dict[str, Any]) -> WholeLLIRCacheEntry:
    return _WHOLE_LLIR_CACHE.put(key, llir, metadata)


def cache_stats() -> dict[str, int]:
    return _WHOLE_LLIR_CACHE.stats()


def replay_metadata(metadata: dict[str, Any], entry: WholeLLIRCacheEntry) -> None:
    metadata.update(entry.metadata)


def stage_event(hit: bool, key: WholeLLIRCacheKey, entry: WholeLLIRCacheEntry | None = None) -> dict[str, Any]:
    stats = cache_stats()
    event: dict[str, Any] = {
        "helioncache_ttgir_llir_enabled": True,
        "helioncache_ttgir_llir_cache_level": cache_level(),
        "helioncache_ttgir_llir_hit": hit,
        "helioncache_whole_llir_hit": hit,
        "ttgir_hash": key.ttgir_hash,
        "ttgir_context_hash": key.context_hash,
        "helioncache_l0_entries": stats["entries"],
        "helioncache_l0_bytes": stats["bytes"],
    }
    if entry is not None:
        event["llir_hash"] = entry.llir_hash
    return event


def _module_str_nodebug(module) -> str:
    str_nodebug = getattr(module, "str_nodebug", None)
    if str_nodebug is not None:
        return str_nodebug()
    get_operation = getattr(module, "get_operation", None)
    if get_operation is not None:
        operation = get_operation()
        str_nodebug = getattr(operation, "str_nodebug", None)
        if str_nodebug is not None:
            return str_nodebug()
    return module.str()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_text_hash(text: str) -> str:
    return _sha256_text(text)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
