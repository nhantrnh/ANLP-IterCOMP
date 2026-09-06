"""Chạy toàn bộ test không cần pytest.

Repo này phải tự chạy được trên máy sạch, và pytest không nằm trong phụ thuộc
lõi. Script này gọi thẳng các hàm test_* để `python tests/run_tests.py` luôn
hoạt động; ai có pytest thì `python -m pytest tests/ -q` vẫn chạy bình thường.
"""
import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    failed = 0
    total = 0
    for path in sorted(HERE.glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            total += 1
            try:
                getattr(mod, name)()
                print(f"  PASS  {path.stem}.{name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {path.stem}.{name}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
