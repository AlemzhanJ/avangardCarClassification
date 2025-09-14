"use client";

import * as ort from "onnxruntime-web";

let sessionPromise: Promise<ort.InferenceSession> | null = null;

const MODEL_URL = "/models/dirt_classifier.onnx";
const DEFAULT_IMG_SIZE = 384;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export type CleanlinessPred = {
  label: "clean" | "dirty";
  confidence: number;
  probabilities: Record<"clean" | "dirty", number>;
};

async function getSession() {
  if (!sessionPromise) {
    const env = (ort as unknown as { env: { wasm: { numThreads?: number; simd?: boolean; wasmPaths?: string } } }).env;
    env.wasm.numThreads = 1;
    env.wasm.simd = true;
    // Self-host WASM binaries to avoid wrong MIME types
    env.wasm.wasmPaths = "/ort/";
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
  }
  return sessionPromise;
}

function softmax(logits: Float32Array): Float32Array {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return new Float32Array(exps.map((v) => v / sum));
}

async function loadAndPreprocess(imageSrc: string, imgSize: number): Promise<ort.Tensor> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = reject;
    i.src = imageSrc;
  });

  const iw = img.naturalWidth || img.width;
  const ih = img.naturalHeight || img.height;
  const side = Math.min(iw, ih);
  const sx = Math.floor((iw - side) / 2);
  const sy = Math.floor((ih - side) / 2);

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context not available");
  canvas.width = imgSize;
  canvas.height = imgSize;

  // Center-crop to square then resize to model size
  ctx.drawImage(img, sx, sy, side, side, 0, 0, imgSize, imgSize);

  const { data } = ctx.getImageData(0, 0, imgSize, imgSize);

  const floatData = new Float32Array(1 * 3 * imgSize * imgSize);
  let idx = 0;
  const stride = imgSize * imgSize;
  for (let y = 0; y < imgSize; y++) {
    for (let x = 0; x < imgSize; x++) {
      const p = (y * imgSize + x) * 4; // RGBA
      const r = data[p] / 255;
      const g = data[p + 1] / 255;
      const b = data[p + 2] / 255;

      floatData[idx] = (r - MEAN[0]) / STD[0];
      floatData[idx + stride] = (g - MEAN[1]) / STD[1];
      floatData[idx + 2 * stride] = (b - MEAN[2]) / STD[2];
      idx++;
    }
  }

  return new ort.Tensor("float32", floatData, [1, 3, imgSize, imgSize]);
}

export async function inferCleanliness(imageSrc: string): Promise<CleanlinessPred> {
  const session = await getSession();
  const inputName = session.inputNames?.[0] ?? "images";
  const outputName = session.outputNames?.[0] ?? "logits";

  type MetaShape = { inputMetadata?: Record<string, { dimensions: (number | string | null)[] }> };
  const meta = (session as unknown as MetaShape).inputMetadata?.[inputName];
  const dims: (number | string | null)[] | undefined = meta?.dimensions;
  const inferredSize =
    Array.isArray(dims) && typeof dims[2] === "number" && typeof dims[3] === "number"
      ? (dims[2] as number)
      : DEFAULT_IMG_SIZE;

  const inputTensor = await loadAndPreprocess(imageSrc, inferredSize);
  const outputs = await session.run({ [inputName]: inputTensor });
  const out = outputs[outputName];
  if (!out) throw new Error("Model output not found");

  const data = out.data as Float32Array | number[];
  const logits = data instanceof Float32Array ? data : new Float32Array(data);
  const probs = softmax(logits);

  // Expect order [clean, dirty]
  const clean = probs[0] ?? 0;
  const dirty = probs[1] ?? 0;
  const label: "clean" | "dirty" = dirty >= clean ? "dirty" : "clean";
  const confidence = Math.max(clean, dirty);

  return {
    label,
    confidence,
    probabilities: { clean, dirty },
  };
}
