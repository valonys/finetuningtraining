/**
 * REST client for the ValonyLabs Studio FastAPI backend.
 *
 * In dev mode, Vite proxies /healthz and /v1/* to localhost:8000 — see
 * vite.config.ts. In production, the frontend is served by the same
 * FastAPI process (or by any static host) so requests go to the same
 * origin.
 */
import type {
  ChatRequest,
  ChatResponse,
  DomainConfigCreate,
  DomainConfigInfo,
  DomainConfigList,
  ForgeBuildResponse,
  HealthResponse,
  JobStatus,
  TrainingJobRequest,
  UploadListResponse,
  UploadResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "";

async function call<T>(
  path: string,
  opts?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(opts?.headers as Record<string, string>) },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.text();
    let detail: string;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {
      detail = body;
    }
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ── Health ────────────────────────────────────────────────────
export const getHealth = () => call<HealthResponse>("/healthz");

// ── Templates & OCR ──────────────────────────────────────────
export const getTemplates = () => call<{ templates: string[] }>("/v1/templates");
export const getOCREngines = () => call<{ engines: string[] }>("/v1/ocr/engines");

// ── Domain configs ───────────────────────────────────────────
export const listDomainConfigs = () => call<DomainConfigList>("/v1/domains/configs");
export const getDomainConfig = (name: string) =>
  call<DomainConfigInfo>(`/v1/domains/configs/${encodeURIComponent(name)}`);
export const createDomainConfig = (body: DomainConfigCreate) =>
  call<DomainConfigInfo>("/v1/domains/configs", {
    method: "POST",
    body: JSON.stringify(body),
  });
export const getDomainTemplate = () => call<{ template: Record<string, unknown> }>("/v1/domains/template");

// ── Data Forge ───────────────────────────────────────────────
export const forgeBuildDataset = (body: {
  paths: string[];
  task: string;
  base_model: string;
  template?: string;
  system_prompt: string;
  synth_qa: boolean;
  target_size: number | null;
  filter_noise?: boolean;
}) =>
  call<ForgeBuildResponse>("/v1/forge/build_dataset", {
    method: "POST",
    body: JSON.stringify(body),
  });

export const forgeHarvestYoutube = (body: {
  query: string;
  max_videos: number;
  min_chars?: number;
}) =>
  call<import("./types").YouTubeHarvestResponse>(
    "/v1/forge/harvest/youtube",
    { method: "POST", body: JSON.stringify(body) }
  );

/**
 * Upload one or more files via multipart/form-data. Do NOT set a
 * Content-Type header — the browser sets it automatically with the
 * correct multipart boundary. Uses XHR (not fetch) so we can surface
 * upload progress, which matters for big PDFs and image batches.
 */
export function forgeUpload(
  files: File[],
  onProgress?: (loaded: number, total: number) => void
): Promise<UploadResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);

  return new Promise<UploadResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/v1/forge/upload`);
    xhr.upload.onprogress = (e) => {
      if (onProgress && e.lengthComputable) onProgress(e.loaded, e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText) as UploadResponse); }
        catch (e) { reject(new Error(`Invalid response: ${e}`)); }
      } else {
        let detail = xhr.responseText;
        try { detail = JSON.parse(xhr.responseText).detail ?? detail; } catch { /* keep raw */ }
        reject(new Error(`${xhr.status} ${xhr.statusText}: ${detail}`));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed (network error)"));
    xhr.send(form);
  });
}

export const forgeListUploads = () => call<UploadListResponse>("/v1/forge/uploads");

export const forgeDeleteUpload = (filename: string) =>
  call<{ deleted: string }>(
    `/v1/forge/uploads/${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );

export const forgeClearUploads = () =>
  call<{ deleted: number }>("/v1/forge/uploads", { method: "DELETE" });

// ── Training jobs ────────────────────────────────────────────
export const createTrainingJob = (body: TrainingJobRequest) =>
  call<JobStatus>("/v1/jobs/create", {
    method: "POST",
    body: JSON.stringify(body),
  });
export const getJob = (jobId: string) => call<JobStatus>(`/v1/jobs/${jobId}`);
export const listJobs = () => call<JobStatus[]>("/v1/jobs");

// ── Trained adapters ─────────────────────────────────────────
export const listTrainedDomains = () =>
  call<{ domain_name: string; adapter_path: string }[]>("/v1/domains");
export const reloadInference = () =>
  call<{ status: string; domains: string[] }>("/v1/inference/reload", { method: "POST" });

// ── Chat / inference ─────────────────────────────────────────
export const chat = (body: ChatRequest) =>
  call<ChatResponse>("/v1/chat", {
    method: "POST",
    body: JSON.stringify(body),
  });

/**
 * Streaming chat over Server-Sent Events.
 *
 * Backend emits frames like:
 *    data: {"delta": "Hello"}
 *    data: {"delta": " world"}
 *    data: {"meta": {backend, model, ttft_ms, tokens_generated, ...}}
 *    data: [DONE]
 *
 * We parse the frames as bytes arrive and invoke the callbacks so the
 * chat widget can show the typewriter effect live.
 *
 * Returns a promise that resolves when the stream closes naturally,
 * rejects on network / parse errors. Pass an AbortSignal to cancel
 * mid-stream (e.g. when the user navigates away).
 */
export async function chatStream(
  body: ChatRequest,
  callbacks: {
    onDelta?: (delta: string) => void;
    onMeta?: (meta: Partial<ChatResponse>) => void;
    onError?: (error: string) => void;
    onSources?: (sources: import("./types").DocsSource[]) => void;
  },
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${BASE}/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  if (!res.body) throw new Error("Response has no body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are `\n\n`-separated. Each frame has one or more
    // `data: ` lines.
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      // Drop `data: ` prefix on each line and join (SSE spec allows
      // multi-line data fields).
      const dataLines = frame
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart());
      if (dataLines.length === 0) continue;
      const payload = dataLines.join("\n");

      if (payload === "[DONE]") return;

      try {
        const obj = JSON.parse(payload) as {
          delta?: string;
          meta?: Partial<ChatResponse>;
          error?: string;
          sources?: import("./types").DocsSource[];
        };
        if (obj.error !== undefined && callbacks.onError) callbacks.onError(obj.error);
        else if (obj.sources !== undefined && callbacks.onSources) callbacks.onSources(obj.sources);
        else if (obj.delta !== undefined && callbacks.onDelta) callbacks.onDelta(obj.delta);
        else if (obj.meta !== undefined && callbacks.onMeta) callbacks.onMeta(obj.meta);
      } catch {
        // Malformed frame — skip silently rather than abort the stream
      }
    }
  }
}
