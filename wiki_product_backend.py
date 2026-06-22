"""
wiki_product_backend.py — 评论写手接入 LLM-Wiki 替代产品信息库

🎯 目标:
    替代 E:/产品信息知识库/crawler/product_retriever.py 的 search() 接口.
    用 deepsearch/wiki/<product>/ (4353 md) 替代 Milvus 'lok_world' 等产品信息库.

📊 替代关系:
    Milvus 6 collections (产品信息库):
        honor_of_kings, honor_of_kings_world, lok_world, jcjz, dnf, wxqy
        (共 ~600 篇文章, 用 SentenceTransformer embedding + COSINE 检索)

    替代 → LLM-Wiki 6 产品 (官方结构化数据):
        wangzhe, wangzhe-world, luoke, jcczz, dnf, valm
        (共 4353 md, ReAct 实体浏览)

🚀 3 种部署模式:
    1. local:  本机直接 import deepsearch.scripts.query (最快, 需本机有 wiki)
    2. http:   调远程 Wiki API Server (推荐, 服务器端部署用此模式)
    3. auto:  优先 http (环境变量配置), 失败降级 local

🔧 用法 (drop-in 替换):
    # 原 RAG:
    from crawler.product_retriever import search as kb_search
    results = kb_search("星光对决 莫扎特", product="lok_world", top_k=10)

    # 替换 Wiki:
    from wiki_product_backend import search as kb_search  # 接口同名
    results = kb_search("星光对决 莫扎特", product="洛克王国世界", top_k=10)

🌐 服务器端部署 (HTTP 模式):
    # 在 .env 或环境变量:
    WIKI_API_URL=http://192.168.x.x:8088
    WIKI_API_KEY=deepsearch123  (可选)
    WIKI_BACKEND=http  (默认 auto, 读 WIKI_API_URL 自动切换)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# === 接入 deepsearch 的 query.py (仅 local 模式需要) ===
_DEEPSEARCH_ROOT = Path(__file__).resolve().parent.parent / "deepsearch"
if str(_DEEPSEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEEPSEARCH_ROOT))

try:
    from scripts.query import query as _wiki_query
    _WIKI_AVAILABLE = True
except ImportError as e:
    _WIKI_AVAILABLE = False
    _IMPORT_ERR = str(e)

# === HTTP backend: 调远程 Wiki API Server ===
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# === Stance → deepsearch wiki product 映射 ===
STANCE_TO_DEEPSEARCH_PRODUCT = {
    "王者荣耀":       "wangzhe",
    "王者荣耀世界":   "wangzhe-world",
    "洛克王国世界":   "luoke",
    "DNF端游":        "dnf",
    "金铲铲之战":     "jcczz",
    "无畏契约手游":   "valm",
}

# === Milvus collection → deepsearch product ===
COLLECTION_TO_DEEPSEARCH_PRODUCT = {
    "honor_of_kings":         "wangzhe",
    "honor_of_kings_world":   "wangzhe-world",
    "lok_world":              "luoke",
    "jcjz":                   "jcczz",
    "dnf":                    "dnf",
    "wxqy":                   "valm",
}


class WikiProductBackend:
    """
    LLM-Wiki 产品信息后端 — drop-in 替代 ProductKnowledgeBase.search

    接口: search(query_text, product=None, top_k=5) -> List[Dict]

    返回 dict 字段 (兼容原 KB):
        product, product_display, id, title, url, source, date,
        content_text, content_length, distance
    """

    _cache: Dict[str, Dict[str, tuple]] = {}
    _CACHE_TTL = 3600  # 1 小时

    # LLM fallback 提示词 — 检测到就视为检索失败
    FALLBACK_MARKERS = [
        "达到 ReAct 步数上限",
        "达到步数上限",
        "ReAct 步数上限",
        "超时",
        "知识库中可能暂无该信息",
        "知识库中暂无",
        "知识库暂未收录",
        "可尝试",
        "请尝试",
        "建议在 wiki",
        "模型未能在限制内定位",
        "未能在限制内定位",
    ]

    def __init__(self, verbose: bool = False, timeout_sec: int = 60, backend: str = "auto"):
        """
        Args:
            verbose: 打印检索过程
            timeout_sec: 单次查询超时
            backend: "local" / "http" / "auto"
                - local: import deepsearch.scripts.query (本机)
                - http:   调远程 Wiki API Server (服务器部署)
                - auto:   读 WIKI_API_URL 环境变量自动切换
        """
        self.verbose = verbose
        self.timeout_sec = timeout_sec
        self.backend = self._resolve_backend(backend)
        self._cascade_writes: set = set()

        if self.backend == "local" and not _WIKI_AVAILABLE:
            raise RuntimeError(
                f"local backend 不可用: {_IMPORT_ERR}\n"
                f"服务器部署请设置 WIKI_API_URL + backend='http'"
            )
        if self.backend == "http" and not _REQUESTS_AVAILABLE:
            raise RuntimeError("http backend 需要 requests 库")

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend != "auto":
            return backend
        if os.getenv("WIKI_API_URL"):
            return "http"
        return "local"

    def search(
        self,
        query_text: str,
        product: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        deepsearch_product = self._resolve_product(product)

        cache_key = self._make_cache_key(deepsearch_product, query_text)
        cached = self._cache_get(cache_key, deepsearch_product)
        if cached is not None:
            if self.verbose:
                print(f"[WikiProduct] cache hit: {query_text[:40]}")
            return cached[:top_k]

        results = self._do_query(query_text, deepsearch_product)
        if self._is_fallback_results(results):
            if self.verbose:
                print(f"[WikiProduct] fallback, retry with simplified query")
            simplified = self._simplify_query(query_text)
            results = self._do_query(simplified, deepsearch_product)
            if self._is_fallback_results(results):
                if self.verbose:
                    print(f"[WikiProduct] still fallback, return []")
                results = []

        self._cache_set(cache_key, deepsearch_product, results)
        return results[:top_k]

    def _do_query(self, query_text: str, product: str) -> List[Dict[str, Any]]:
        if self.backend == "http":
            return self._do_query_http(query_text, product)
        return self._do_query_local(query_text, product)

    def _do_query_local(self, query_text: str, product: str) -> List[Dict[str, Any]]:
        """本地 LLM 查询"""
        try:
            t0 = time.time()
            if self.verbose:
                print(f"[WikiProduct:local] query: {query_text[:60]} (product={product})")
            answer = _wiki_query(query_text, verbose=False, product=product)
            elapsed = time.time() - t0
            return self._answer_to_results(answer, query_text, product, elapsed)
        except Exception as e:
            if self.verbose:
                print(f"[WikiProduct:local] query FAILED: {e}")
            return []

    def _do_query_http(self, query_text: str, product: str) -> List[Dict[str, Any]]:
        """HTTP 调远程 Wiki API Server"""
        url = os.getenv("WIKI_API_URL", "").rstrip("/")
        if not url:
            if self.verbose:
                print(f"[WikiProduct:http] WIKI_API_URL 未配置, fallback to local")
            return self._do_query_local(query_text, product)

        api_key = os.getenv("WIKI_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "query": query_text,
            "product": product,
            "top_k": 10,
            "verbose": False,
        }

        try:
            t0 = time.time()
            if self.verbose:
                print(f"[WikiProduct:http] POST {url}/api/wiki/search (product={product})")
            resp = _requests.post(
                f"{url}/api/wiki/search",
                json=payload,
                headers=headers,
                timeout=self.timeout_sec,
            )
            elapsed = time.time() - t0
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if self.verbose:
                print(f"[WikiProduct:http] {len(results)} hits ({elapsed:.2f}s)")
            return results
        except _requests.Timeout:
            if self.verbose:
                print(f"[WikiProduct:http] TIMEOUT after {self.timeout_sec}s")
            return []
        except Exception as e:
            if self.verbose:
                print(f"[WikiProduct:http] FAILED: {e}")
            return []

    @staticmethod
    def _resolve_product(product: Optional[str]) -> str:
        if not product:
            return "wangzhe"
        if product in STANCE_TO_DEEPSEARCH_PRODUCT:
            return STANCE_TO_DEEPSEARCH_PRODUCT[product]
        if product in COLLECTION_TO_DEEPSEARCH_PRODUCT:
            return COLLECTION_TO_DEEPSEARCH_PRODUCT[product]
        return product

    def _is_fallback_results(self, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return True
        all_text = " ".join(r.get("content_text", "") for r in results)
        return any(m in all_text for m in self.FALLBACK_MARKERS)

    def _simplify_query(self, query_text: str) -> str:
        return f"查询: {query_text[:30]} (简要回答)"

    def _answer_to_results(
        self, answer: str, query_text: str, product: str, elapsed: float
    ) -> List[Dict[str, Any]]:
        if not answer:
            return []
        import re
        paragraphs = [p.strip() for p in re.split(r"\n{2,}|(?<=[。！？])\s+", answer) if p.strip()]
        if not paragraphs:
            paragraphs = [answer]
        results = []
        for i, para in enumerate(paragraphs):
            results.append({
                "product": product,
                "product_display": product,
                "id": f"wiki/{product}/llm/{i}",
                "title": f"参考片段 #{i+1}: {query_text[:30]}",
                "url": f"wiki://deepsearch/{product}/(LLM 综合)",
                "source": "LLM-Wiki",
                "date": time.strftime("%Y-%m-%d"),
                "content_text": para[:500],
                "content_length": len(para),
                "distance": round(0.1 + (i / max(len(paragraphs), 1)) * 0.5, 3),
            })
        return results

    def _make_cache_key(self, product: str, query_text: str) -> str:
        return f"{product}|{query_text[:200]}"

    def _wiki_mtime(self, product: str) -> float:
        """local 模式用, http 模式不查本地"""
        if self.backend != "local":
            return 0.0
        wiki_dir = _DEEPSEARCH_ROOT / "wiki" / product
        if not wiki_dir.exists():
            return 0.0
        max_mtime = 0.0
        for md in wiki_dir.rglob("*.md"):
            try:
                mt = md.stat().st_mtime
                if mt > max_mtime:
                    max_mtime = mt
            except OSError:
                pass
        return max_mtime

    def _cache_get(self, key: str, product: str) -> Optional[List[Dict[str, Any]]]:
        bucket = self._cache.setdefault("default", {})
        if key in bucket:
            cached_ts, cached_mtime, result = bucket[key]
            if time.time() - cached_ts > self._CACHE_TTL:
                return None
            if self._wiki_mtime(product) > cached_mtime + 1:
                return None
            return result
        return None

    def _cache_set(self, key: str, product: str, result: List[Dict[str, Any]]):
        bucket = self._cache.setdefault("default", {})
        bucket[key] = (time.time(), self._wiki_mtime(product), result)

    def list_collections(self) -> List[Dict[str, Any]]:
        return [
            {"collection": v, "display_name": k, "article_count": "N/A", "status": "wiki"}
            for k, v in STANCE_TO_DEEPSEARCH_PRODUCT.items()
        ]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "deepsearch_root": str(_DEEPSEARCH_ROOT),
            "wiki_api_url": os.getenv("WIKI_API_URL", ""),
        }


# === 顶层 search 函数 — drop-in 替代 product_retriever.search ===
_default_backend: Optional[WikiProductBackend] = None

def search(
    query_text: str,
    product: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    global _default_backend
    if _default_backend is None:
        _default_backend = WikiProductBackend()
    return _default_backend.search(query_text, product=product, top_k=top_k)


def list_products() -> List[Dict[str, Any]]:
    if _default_backend is None:
        WikiProductBackend()
    return _default_backend.list_collections()


def get_stats() -> Dict[str, Any]:
    if _default_backend is None:
        WikiProductBackend()
    return _default_backend.get_stats()


# === 自测 ===
if __name__ == "__main__":
    import sys
    # 可选: WIKI_API_URL=http://localhost:8088 WIKI_API_KEY=xxx python wiki_product_backend.py
    backend = WikiProductBackend(verbose=True)
    print(f"backend = {backend.backend}")

    tests = [
        ("星光对决 莫扎特", "洛克王国世界"),
        ("李白的 1 技能", "王者荣耀"),
    ]

    for query, product in tests:
        print(f"\n=== {product} | {query[:30]} ===")
        results = backend.search(query, product=product, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] d={r['distance']:.3f} | {r['title'][:30]}")
            print(f"      {r['content_text'][:100]}...")
