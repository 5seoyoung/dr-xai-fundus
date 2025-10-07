# scripts/make_cam_board.py
import argparse, io, random
from pathlib import Path
import cv2, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from src.preprocess import preprocess_fundus_path

def pick_balanced(norm_paths, dise_paths, k_each=6, seed=42):
    random.seed(seed)
    n_pick = min(len(norm_paths), k_each)
    d_pick = min(len(dise_paths), k_each)
    norm_sel = random.sample(norm_paths, n_pick) if n_pick>0 else []
    dise_sel = random.sample(dise_paths, d_pick) if d_pick>0 else []
    if n_pick < k_each and len(dise_paths) > d_pick:
        dise_sel += random.sample([p for p in dise_paths if p not in dise_sel], k_each-n_pick)
    if d_pick < k_each and len(norm_paths) > n_pick:
        norm_sel += random.sample([p for p in norm_paths if p not in norm_sel], k_each-d_pick)
    return norm_sel + dise_sel

def save_board(paths, aptos_dir, title, out_jpg, img_size=448, dpi=220):
    if not paths:
        print(f"[skip] {title}: no images"); return
    cols = 4
    rows = int(np.ceil(len(paths)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes: ax.axis('off')
    for i, p in enumerate(paths[:rows*cols]):
        name = p.name.replace("val_", "")
        img = preprocess_fundus_path(aptos_dir/name, size=img_size)
        if img is None: continue
        cam = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        panel = np.hstack([img, cam])
        axes[i].imshow(panel); axes[i].set_title(name, fontsize=8); axes[i].axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.97])
    out = Path(out_jpg); out.parent.mkdir(parents=True, exist_ok=True)
    try:
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=dpi); plt.close(fig)
        buf.seek(0); Image.open(buf).convert("RGB").save(out, format="JPEG", quality=90); buf.close()
        print("Saved:", out)
    except Exception as e:
        print("[warn] save failed -> showing only:", e)
        plt.show()

def main(args):
    cams_dir = Path(args.cams_dir)
    aptos_dir = Path(args.aptos_images)
    cam_paths = sorted(cams_dir.glob("val_*.png"))
    if not cam_paths:
        raise FileNotFoundError(f"No 'val_*.png' under {cams_dir}")

    if args.val_csv:
        df = pd.read_csv(args.val_csv)
        df["stem"] = df["image_file"].astype(str).str.replace(".png","",regex=False).str.replace(".jpg","",regex=False)
        label_by_stem = dict(zip(df["stem"], df["binary"].astype(int)))
        normals, diseased = [], []
        for p in cam_paths:
            stem = p.stem.replace("val_","")
            lb = label_by_stem.get(stem)
            if lb is None: continue
            (normals if lb==0 else diseased).append(p)
        picked = pick_balanced(normals, diseased, k_each=6, seed=args.seed)
        title = "MIXED (Val) — Original vs CAM (6 Normal + 6 Diseased)"
    else:
        random.seed(args.seed)
        picked = random.sample(cam_paths, min(12, len(cam_paths)))
        title = "MIXED (Val) — Original vs CAM (12 samples)"

    save_board(picked, aptos_dir, title, args.out_jpg, img_size=args.img_size, dpi=args.dpi)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams_dir", required=True, help=".../cams_resume")
    ap.add_argument("--aptos_images", required=True, help=".../train_images")
    ap.add_argument("--val_csv", default="", help="optional df_val csv with image_file,binary")
    ap.add_argument("--out_jpg", default="artifacts/figs/board_mixed.jpg")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
