# src/preprocess.py
import cv2, numpy as np

def make_retina_mask(rgb):
    g = rgb[:, :, 1]
    g_blur = cv2.medianBlur(g, 9)
    _, th = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones(g.shape, np.uint8) * 255
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(g, np.uint8); cv2.drawContours(mask, [c], -1, 255, -1)
    (x, y), r = cv2.minEnclosingCircle(c)
    circ = np.zeros_like(g, np.uint8); cv2.circle(circ, (int(x), int(y)), int(r*0.98), 255, -1)
    return cv2.bitwise_and(mask, circ)

def apply_clahe(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)

def emphasize_green(rgb):
    r, g, b = cv2.split(rgb)
    mix = cv2.addWeighted(r, 0.1, g, 0.8, 0)
    mix = cv2.addWeighted(mix, 1.0, b, 0.1, 0)
    return cv2.merge([mix, mix, mix])

def resize_pad_square(rgb, size=448):
    h, w = rgb.shape[:2]; s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    pt = (size - nh) // 2; pb = size - nh - pt
    pl = (size - nw) // 2; pr = size - nw - pl
    return cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REFLECT_101)

def preprocess_fundus_path(path, size=448, use_mask=True, use_clahe=True, use_green=True):
    bgr = cv2.imread(str(path))
    if bgr is None: return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if use_mask:
        mask = make_retina_mask(rgb); rgb[mask == 0] = 0
    if use_clahe:
        rgb = apply_clahe(rgb)
    if use_green:
        rgb = emphasize_green(rgb)
    return resize_pad_square(rgb, size)
