"""M0 验收:验证 Python 能连上 Ollama 与 Qdrant。

用法: python scripts/healthcheck.py
"""
import sys
import httpx
from rag.config import get_settings


def check_ollama(settings) -> bool:
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/version", timeout=5)
        r.raise_for_status()
        print(f"✅ Ollama 通了 —— version={r.json().get('version')}")
        return True
    except Exception as e:
        print(f"❌ Ollama 连不上 ({settings.ollama_base_url}): {e}")
        return False


def check_qdrant(settings) -> bool:
    try:
        r = httpx.get(f"{settings.qdrant_url}/healthz", timeout=5)
        r.raise_for_status()
        print(f"✅ Qdrant 通了 —— {settings.qdrant_url}")
        return True
    except Exception as e:
        print(f"❌ Qdrant 连不上 ({settings.qdrant_url}): {e}")
        return False


def main() -> int:
    settings = get_settings()
    ok_ollama = check_ollama(settings)
    ok_qdrant = check_qdrant(settings)
    if ok_ollama and ok_qdrant:
        print("\n🎉 M0 验收通过:环境就绪!")
        return 0
    print("\n⚠️  有服务未就绪,请检查 Ollama / Docker 是否启动。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
