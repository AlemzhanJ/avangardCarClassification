"use client";

import * as ort from "onnxruntime-web";

// Cache session across calls
let sessionPromise: Promise<ort.InferenceSession> | null = null;

const MODEL_URL = "/models/severity_classifier.onnx";
// Default; will be overridden by model metadata if available
const DEFAULT_IMG_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export type SeverityPred = {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  severity: "low" | "med" | "high";
  damage_type: "scratch" | "dent";
};

const CLASS_NAMES = [
  "scratch_low",
  "scratch_med",
  "scratch_high",
  "dent_low",
  "dent_med",
  "dent_high",
] as const;

async function getSession() {
  if (!sessionPromise) {
    // Configure runtime to avoid threaded build (no COOP/COEP required on Vercel)
    const env = (ort as unknown as { env: { wasm: { numThreads?: number; simd?: boolean; wasmPaths?: string } } }).env;
    env.wasm.numThreads = 1;
    env.wasm.simd = true;
    // Self-host WASM files to ensure correct MIME type
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

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context not available");
  canvas.width = imgSize;
  canvas.height = imgSize;

  // Draw with simple resize (cover/fill could be added if needed)
  ctx.drawImage(img, 0, 0, imgSize, imgSize);

  const { data } = ctx.getImageData(0, 0, imgSize, imgSize);

  // Convert to shape [1,3,384,384] RGB float32 normalized
  const floatData = new Float32Array(1 * 3 * imgSize * imgSize);
  let idx = 0;
  const stride = imgSize * imgSize;
  for (let y = 0; y < imgSize; y++) {
    for (let x = 0; x < imgSize; x++) {
      const p = (y * imgSize + x) * 4; // RGBA
      const r = data[p] / 255;
      const g = data[p + 1] / 255;
      const b = data[p + 2] / 255;

      // CHW layout
      floatData[idx] = (r - MEAN[0]) / STD[0]; // R
      floatData[idx + stride] = (g - MEAN[1]) / STD[1]; // G
      floatData[idx + 2 * stride] = (b - MEAN[2]) / STD[2]; // B
      idx++;
    }
  }

  return new ort.Tensor("float32", floatData, [1, 3, imgSize, imgSize]);
}

export async function inferSeverity(imageSrc: string): Promise<SeverityPred> {
  const session = await getSession();

  // Resolve input/output names generically to avoid mismatches
  const inputName = session.inputNames?.[0] ?? "input";
  const outputName = session.outputNames?.[0] ?? "logits";

  // Derive expected spatial size from model metadata if available
  type MetaShape = { inputMetadata?: Record<string, { dimensions: (number | string | null)[] }> };
  const meta = (session as unknown as MetaShape).inputMetadata?.[inputName];
  const dims: (number | string | null)[] | undefined = meta?.dimensions;
  const inferredSize =
    Array.isArray(dims) && typeof dims[2] === "number" && typeof dims[3] === "number"
      ? (dims[2] as number)
      : DEFAULT_IMG_SIZE;

  const inputTensor = await loadAndPreprocess(imageSrc, inferredSize);

  const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
  const results = await session.run(feeds);
  const output = results[outputName];
  if (!output) throw new Error("Model output not found");

  const data = output.data as Float32Array | number[];
  const logits = data instanceof Float32Array ? data : new Float32Array(data);
  const probs = softmax(logits);

  // Argmax
  let maxIdx = 0;
  let maxVal = -Infinity;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > maxVal) {
      maxVal = probs[i];
      maxIdx = i;
    }
  }

  // Map to labels; clamp if model has different size
  const idx = Math.min(maxIdx, CLASS_NAMES.length - 1);
  const predictedClass = CLASS_NAMES[idx];

  const probabilities: Record<string, number> = {};
  for (let i = 0; i < CLASS_NAMES.length && i < probs.length; i++) {
    probabilities[CLASS_NAMES[i]] = probs[i];
  }

  const [damage_type, severity] = predictedClass.split("_") as [
    "scratch" | "dent",
    "low" | "med" | "high"
  ];

  return {
    predicted_class: predictedClass,
    confidence: probs[maxIdx] ?? 0,
    probabilities,
    severity,
    damage_type,
  };
}
