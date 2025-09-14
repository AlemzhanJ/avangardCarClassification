"use client";

import * as ort from "onnxruntime-web";

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let labelsPromise: Promise<Record<number, string>> | null = null;

const MODEL_URL = "/models/severity_classifier.onnx";
const LABELS_URL = "/models/severity_classifier.labels.json";
const DEFAULT_IMG_SIZE = 384;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export type SeverityPred = {
  label: "undamaged" | "damaged";
  confidence: number;
  probabilities: Record<"undamaged" | "damaged", number>;
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
      .then((data) => {
        const result: Record<number, string> = {};
        for (const k of Object.keys(data)) {
          const idx = Number(k);
          if (!Number.isNaN(idx)) result[idx] = String((data as any)[k]);
        }
        return result;
      })
      .catch(() => ({}));
  }
  return labelsPromise;
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
  ctx.drawImage(img, 0, 0, imgSize, imgSize);

  const { data } = ctx.getImageData(0, 0, imgSize, imgSize);

  const floatData = new Float32Array(1 * 3 * imgSize * imgSize);
  let idx = 0;
  const stride = imgSize * imgSize;
  for (let y = 0; y < imgSize; y++) {
    for (let x = 0; x < imgSize; x++) {
      const p = (y * imgSize + x) * 4;
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

export async function inferSeverity(imageSrc: string): Promise<SeverityPred> {
  const session = await getSession();
  const idxToClass = await getLabels();
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

  // Resolve indices from labels mapping if available, else fallback
  let damagedIdx = -1, undamagedIdx = -1;
  for (const [k, v] of Object.entries(idxToClass)) {
    if (v === 'damaged') damagedIdx = Number(k);
    if (v === 'undamaged') undamagedIdx = Number(k);
  }
  if (damagedIdx < 0) damagedIdx = 0; // default to [damaged, undamaged]
  if (undamagedIdx < 0) undamagedIdx = 1;

  const damaged = probs[damagedIdx] ?? 0;
  const undamaged = probs[undamagedIdx] ?? 0;
  const label: "undamaged" | "damaged" = damaged >= undamaged ? "damaged" : "undamaged";
  const confidence = Math.max(undamaged, damaged);

  return {
    label,
    confidence,
    probabilities: { undamaged, damaged },
  };
}
