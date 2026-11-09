"""Kronos のモデル本体（https://github.com/shiyu-coder/Kronos の model/ をそのまま取り込んだもの）。

取り込んだ版: 67b630e（2026-04-13）。MIT License（LICENSE を参照）。
変更点は kronos.py の2行だけ: `from model.module import *` を相対 import にし、`sys.path.append("../")` を消した。
torch が要るので、使うときは `pip install -e ".[kronos]"`。
"""

from .kronos import Kronos, KronosPredictor, KronosTokenizer

__all__ = ["Kronos", "KronosPredictor", "KronosTokenizer"]
