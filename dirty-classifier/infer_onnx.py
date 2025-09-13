#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort


def pick_providers() -> List[str]:
    avail = set(ort.get_available_providers())
    # Prefer CoreML on Apple, then CPU
    if 'CoreMLExecutionProvider' in avail:
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def load_labels(model_path: Path) -> List[str]:
    # Try model.labels.json next to ONNX
    labels_path = model_path.with_suffix('.labels.json')
    if labels_path.exists():
        labels = json.load(open(labels_path))
        # Keys may be strings of ints
        return [labels[str(i)] for i in range(len(labels))]

    # Try dataset_meta.json one level up (as in our training outputs)
    meta_path = model_path.parent / 'dataset_meta.json'
    if meta_path.exists():
        meta = json.load(open(meta_path))
        idx2 = meta.get('idx_to_class', {"0": "clean", "1": "dirty"})
        return [idx2[str(i)] for i in sorted(map(int, idx2.keys()))]

    return ['clean', 'dirty']


def preprocess(img_path: Path, img_size: int) -> np.ndarray:
    # Match eval pipeline: Resize(1.15x) -> CenterCrop -> ToTensor -> Normalize(ImageNet)
    im = Image.open(img_path).convert('RGB')
    im = ImageOps.exif_transpose(im)
    short = int(img_size * 1.15)
    scale = short / min(im.size)
    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BICUBIC)
    left = max(0, (im.width - img_size) // 2)
    top = max(0, (im.height - img_size) // 2)
    im = im.crop((left, top, left + img_size, top + img_size))

    x = np.asarray(im).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, ...]  # NCHW with batch dim
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run():
    ap = argparse.ArgumentParser(description='Infer ONNX clean/dirty on image or folder')
    ap.add_argument('--model', required=True, help='Path to ONNX model file')
    ap.add_argument('--image', help='Path to a single image')
    ap.add_argument('--dir', help='Directory with images (jpg/png/jpeg/webp/bmp)')
    ap.add_argument('--img-size', type=int, default=384, help='Input image size (default: 384)')
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    classes = load_labels(model_path)
    providers = pick_providers()
    sess = ort.InferenceSession(model_path.as_posix(), providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    def infer_file(p: Path):
        x = preprocess(p, args.img_size)
        logits = sess.run([output_name], {input_name: x})[0]
        probs = softmax(logits)[0]
        pred_idx = int(probs.argmax())
        pred_cls = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
        print(f"{p}: {pred_cls} | probs: " + ", ".join(f"{c}={probs[i]:.4f}" for i, c in enumerate(classes)))

    if args.image:
        infer_file(Path(args.image))
    elif args.dir:
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        files = [p for p in Path(args.dir).glob('*') if p.suffix.lower() in exts]
        files.sort()
        if not files:
            print('No images found in directory.')
        for p in files:
            infer_file(p)
    else:
        raise SystemExit('Provide --image or --dir')


if __name__ == '__main__':
    run()

