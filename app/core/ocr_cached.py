from app.schemas.chart_analysis import OcrBox
import easyocr
import numpy as np
from typing import List


_reader = None

def get_reader(langs=("en",)):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(list(langs), gpu=False)
    return _reader

def run_ocr(image_bgr: np.ndarray, langs=("en",)) -> List[OcrBox]:
    rdr = get_reader(langs)
    img_rgb = image_bgr[:, :, ::-1]
    res = rdr.readtext(img_rgb, detail=1, paragraph=False)
    out = []
    for bbox, txt, conf in res:
        t = (txt or "").strip()
        if not t:
            continue
        out.append(OcrBox(text=t, conf=float(conf), bbox=[[float(x), float(y)] for x, y in bbox]))
    return out
