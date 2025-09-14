#!/usr/bin/env python3
"""
Export a YOLO model (.pt) to ONNX.

Defaults:
- weights: yolo/yolo.pt
- out: yolo/yolo.onnx
- imgsz: 640
- opset: 17
- dynamic shapes enabled

Requires: ultralytics>=8.x installed in the active environment.
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO .pt to ONNX")
    p.add_argument("--weights", type=str, default="yolo/yolo.pt", help="Path to YOLO .pt weights")
    p.add_argument("--out", type=str, default=None, help="Output .onnx path (default: same dir/name)")
    p.add_argument("--imgsz", type=int, nargs="*", default=[640], help="Image size: single int or H W")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--dynamic", action="store_true", default=True, help="Enable dynamic input shapes")
    p.add_argument("--no-dynamic", dest="dynamic", action="store_false", help="Disable dynamic shapes")
    p.add_argument("--simplify", action="store_true", help="Simplify ONNX graph (needs onnxsim)")
    p.add_argument("--half", action="store_true", help="Export in FP16 where supported")
    p.add_argument("--nms", action="store_true", help="Embed NMS into ONNX graph (end-to-end)")
    p.add_argument("--device", type=str, default="cpu", help="Device for loading: cpu or cuda")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(
            "Ultralytics is required. Please install it in your venv:\n"
            "  pip install ultralytics\n"
            f"Import error: {e}"
        )

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    # imgsz can be 1 value or 2 values (h, w)
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
    elif len(args.imgsz) == 2:
        imgsz = (args.imgsz[0], args.imgsz[1])
    else:
        raise SystemExit("--imgsz expects one value (square) or two values (H W)")

    # Load model
    model = YOLO(weights_path.as_posix())

    # Determine output path: ultralytics returns the path used; we also allow override
    out_override = Path(args.out).resolve() if args.out else None

    # Export
    print("Exporting to ONNX...")
    exported_path = model.export(
        format="onnx",
        opset=args.opset,
        imgsz=imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
        device=args.device,
        nms=args.nms,
    )

    exported_path = Path(exported_path).resolve()

    # If user provided --out, move/rename the produced file
    if out_override and out_override != exported_path:
        out_override.parent.mkdir(parents=True, exist_ok=True)
        out_override.write_bytes(exported_path.read_bytes())
        exported_path = out_override

    # Save labels mapping next to ONNX
    try:
        import json
        names = getattr(model, "names", None)
        if names is None:
            names = getattr(getattr(model, "model", object()), "names", None)
        if names:
            # ensure dict[int->str]
            if isinstance(names, list):
                names = {i: n for i, n in enumerate(names)}
            labels_path = exported_path.with_suffix('.labels.json')
            with open(labels_path, 'w') as f:
                json.dump(names, f, ensure_ascii=False, indent=2)
            print(f"Labels JSON: {labels_path}")
    except Exception as _:
        pass

    print(f"ONNX saved to: {exported_path}")
    print("Notes:")
    print("- Input: float32 N x 3 x H x W, RGB, 0..1 normalized by Ultralytics preprocessing.")
    if args.nms:
        print("- NMS is embedded: model outputs final detections [x1,y1,x2,y2,conf,class].")
    else:
        print("- Raw predictions: apply decode + NMS outside the model if needed.")


if __name__ == "__main__":
    main()
