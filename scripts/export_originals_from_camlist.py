# scripts/export_originals_from_camlist.py
import argparse
from pathlib import Path
import cv2
from src.preprocess import preprocess_fundus_path

def main(args):
    aptos = Path(args.aptos_images)
    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)

    names = []
    if args.cam_list:
        with open(args.cam_list) as f:
            names = [ln.strip() for ln in f if ln.strip()]
    elif args.cams_dir:
        names = [p.name for p in sorted(Path(args.cams_dir).glob("val_*.png"))[:12]]
    else:
        raise ValueError("Provide --cam_list (txt) or --cams_dir")

    for fname in names:
        stem = fname.replace("val_", "")
        img_path = aptos / stem
        if not img_path.exists():
            print("missing:", img_path); continue
        rgb = preprocess_fundus_path(img_path, size=args.img_size,
                                     use_mask=not args.no_mask,
                                     use_clahe=not args.no_clahe_off,
                                     use_green=not args.no_green_off)
        out = save / fname  # CAM과 동일 이름
        cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("Saved:", out)
    print("Done:", save)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aptos_images", required=True)
    ap.add_argument("--save_dir", default="artifacts/originals_resume")
    ap.add_argument("--cams_dir", default="")
    ap.add_argument("--cam_list", default="")   # txt with lines like 'val_xxx.png'
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--no_clahe_off", action="store_true")
    ap.add_argument("--no_green_off", action="store_true")
    main(ap.parse_args())
