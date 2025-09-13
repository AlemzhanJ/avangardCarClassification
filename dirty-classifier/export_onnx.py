#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

# Reuse model builder from train.py
try:
    from train import build_model
except Exception as e:
    raise RuntimeError(f"Failed to import build_model from train.py: {e}")


def export(checkpoint_dir: Path, out_path: Path, img_size: int | None, opset: int, dynamic_batch: bool):
    checkpoint_dir = checkpoint_dir.resolve()
    out_path = out_path.resolve()

    cfg_path = checkpoint_dir / 'config.json'
    meta_path = checkpoint_dir / 'dataset_meta.json'
    best_path = checkpoint_dir / 'best.pt'
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt not found in {checkpoint_dir}")

    with open(cfg_path) as f:
        cfg_json = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    model_name = cfg_json.get('model', 'resnet50')
    use_pretrained = False  # not needed; weights will be loaded
    img_size = img_size or cfg_json.get('img_size', 384)

    # Build model and load weights
    dummy_cfg = SimpleNamespace(model=model_name, use_pretrained=use_pretrained)
    model = build_model(dummy_cfg, num_classes=2)
    state = torch.load(best_path, map_location='cpu')
    model.load_state_dict(state['model_state'])
    model.eval().to('cpu')

    # Dummy input
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    # Export
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_names = ["images"]
    output_names = ["logits"]
    dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None

    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Save labels mapping next to model
    labels_path = out_path.with_suffix('.labels.json')
    with open(labels_path, 'w') as f:
        json.dump(meta.get('idx_to_class', {0: 'clean', 1: 'dirty'}), f, indent=2)

    print(f"Exported ONNX to: {out_path}")
    print(f"Labels JSON: {labels_path}")
    print("Input: float32 tensor N x 3 x {img} x {img} in [0,1], normalized by ImageNet mean/std.")


def parse_args():
    p = argparse.ArgumentParser(description='Export best checkpoint to ONNX')
    p.add_argument('--checkpoint-dir', type=str, required=True, help='Output dir used during training (contains best.pt)')
    p.add_argument('--out', type=str, default=None, help='ONNX output path; default: <checkpoint-dir>/model.onnx')
    p.add_argument('--img-size', type=int, default=None, help='Input size; defaults to training img_size')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--dynamic-batch', action='store_true', help='Make batch dimension dynamic')
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    out_path = Path(args.out) if args.out else ckpt_dir / 'model.onnx'
    export(ckpt_dir, out_path, args.img_size, args.opset, args.dynamic_batch)


if __name__ == '__main__':
    main()

