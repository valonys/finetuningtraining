"""PaddleOCR backend — great accuracy, CUDA-accelerated on Linux."""
from __future__ import annotations

import numpy as np

from .base import OCREngine, OCRResult


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def __init__(self, lang: str = "en", use_gpu: bool | None = None):
        from paddleocr import PaddleOCR
        if use_gpu is None:
            use_gpu = _cuda_ok()
        self._paddle = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

    @classmethod
    def available(cls) -> bool:
        try:
            import paddleocr   # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, image) -> OCRResult:
        arr = np.array(image.convert("RGB"))
        out = self._paddle.ocr(arr, cls=True)

        blocks: list[dict] = []
        texts: list[str] = []
        confs: list[float] = []

        for page in out or []:
            for line in page or []:
                bbox, (txt, conf) = line
                if not txt:
                    continue
                texts.append(txt)
                confs.append(float(conf))
                blocks.append({
                    "text": txt,
                    "bbox": _flatten_bbox(bbox),
                    "conf": float(conf),
                })

        return OCRResult(
            text="\n".join(texts),
            engine=self.name,
            confidence=float(np.mean(confs)) if confs else 0.0,
            blocks=blocks,
        )


def _flatten_bbox(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return [min(xs), min(ys), max(xs), max(ys)]


def _cuda_ok() -> bool:
    try:
        import paddle
        return paddle.is_compiled_with_cuda()
    except Exception:
        return False
