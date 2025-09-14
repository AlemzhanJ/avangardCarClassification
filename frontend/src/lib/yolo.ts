"use client";

import * as ort from "onnxruntime-web";

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let labelsPromise: Promise<Record<number, string>> | null = null;

const MODEL_URL = "/models/yolo.onnx";
const LABELS_URL = "/models/yolo.labels.json";
const DEFAULT_IMG_SIZE = 640;

export type YoloDet = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  classId: number;
  className?: string;
};

async function getSession() {
  if (!sessionPromise) {
    const env = (ort as unknown as { env: { wasm: { numThreads?: number; simd?: boolean; wasmPaths?: string } } }).env;
    env.wasm.numThreads = 1;
    env.wasm.simd = true;
    env.wasm.wasmPaths = "/ort/";
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
  }
  return sessionPromise;
}

async function getLabels(): Promise<Record<number, string>> {
  if (!labelsPromise) {
    labelsPromise = fetch(LABELS_URL)
      .then(r => (r.ok ? r.json() : Promise.reject(new Error("labels not found"))))
      .then((data: unknown) => {
        const obj = (data ?? {}) as Record<string, unknown>;
        const result: Record<number, string> = {};
        for (const k of Object.keys(obj)) {
          const idx = Number(k);
          if (Number.isNaN(idx)) continue;
          const value = obj[k];
          result[idx] = typeof value === 'string' ? value : String(value);
        }
        return result;
      })
      .catch(() => ({}));
  }
  return labelsPromise;
}

// Direct resize (stretch) full image to square imgSize x imgSize
async function resizeToSquare(imageSrc: string, imgSize: number) {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = reject;
    i.src = imageSrc;
  });

  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;

  const canvas = document.createElement("canvas");
  canvas.width = imgSize;
  canvas.height = imgSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context not available");
  // Stretch original image to square target
  ctx.drawImage(img, 0, 0, iw, ih, 0, 0, imgSize, imgSize);

  // Convert to CHW float32 [0,1]
  const { data } = ctx.getImageData(0, 0, imgSize, imgSize);
  const floatData = new Float32Array(1 * 3 * imgSize * imgSize);
  let idx = 0;
  const stride = imgSize * imgSize;
  for (let y = 0; y < imgSize; y++) {
    for (let x = 0; x < imgSize; x++) {
      const p = (y * imgSize + x) * 4; // RGBA
      const rch = data[p] / 255;
      const gch = data[p + 1] / 255;
      const bch = data[p + 2] / 255;
      floatData[idx] = rch;
      floatData[idx + stride] = gch;
      floatData[idx + 2 * stride] = bch;
      idx++;
    }
  }

  const tensor = new ort.Tensor("float32", floatData, [1, 3, imgSize, imgSize]);
  // Scales to map back from model space (imgSize) to original image
  const scaleX = iw / imgSize;
  const scaleY = ih / imgSize;
  return { tensor, iw, ih, scaleX, scaleY };
}

export async function inferYolo(imageSrc: string, opts?: { confThreshold?: number }): Promise<YoloDet[]> {
  const session = await getSession();
  const labels = await getLabels();

  const inputName = session.inputNames?.[0] ?? "images";
  const outputName = session.outputNames?.[0] ?? session.outputNames?.[0] ?? "output0";

  // Determine target size from metadata if present
  const meta = (session as unknown as { inputMetadata?: Record<string, { dimensions: (number | string | null)[] }> }).inputMetadata?.[inputName];
  const dims = meta?.dimensions;
  const size = Array.isArray(dims) && typeof dims[2] === "number" ? (dims[2] as number) : DEFAULT_IMG_SIZE;

  const { tensor, iw, ih, scaleX, scaleY } = await resizeToSquare(imageSrc, size);

  const outputs = await session.run({ [inputName]: tensor });
  const out = outputs[outputName];
  if (!out) throw new Error("YOLO output not found");

  // Expect [1, N, 6] -> [x1,y1,x2,y2,conf,cls]
  const data = out.data as Float32Array | number[];
  const arr = data instanceof Float32Array ? data : new Float32Array(data);

  const n = (out.dims?.[1] as number) || (arr.length / 6);
  const dets: YoloDet[] = [];
  const confThresh = opts?.confThreshold ?? 0.25;

  for (let i = 0; i < n; i++) {
    const base = i * 6;
    const x1 = arr[base + 0];
    const y1 = arr[base + 1];
    const x2 = arr[base + 2];
    const y2 = arr[base + 3];
    const conf = arr[base + 4];
    const clsId = arr[base + 5] | 0;
    if (!(conf > confThresh)) continue;

    // Map back from stretched square to original image pixels
    const bx1 = Math.max(0, Math.min(iw, x1 * scaleX));
    const by1 = Math.max(0, Math.min(ih, y1 * scaleY));
    const bx2 = Math.max(0, Math.min(iw, x2 * scaleX));
    const by2 = Math.max(0, Math.min(ih, y2 * scaleY));

    dets.push({
      x1: bx1,
      y1: by1,
      x2: bx2,
      y2: by2,
      confidence: conf,
      classId: clsId,
      className: labels[clsId],
    });
  }

  return dets;
}
