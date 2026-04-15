/** Mirrors the Pydantic models in app/models.py. */

export interface HealthResponse {
  status: string;
  version: string;
  hardware: Record<string, unknown>;
  profile: Record<string, unknown>;
  registered_domains: string[];
  inference_backend: string;
  latency_stats: Record<string, number>;
  available_ocr: string[];
  available_templates: string[];
  synth_provider: {
    provider: string;
    model: string | null;
    base_url: string | null;
    is_cloud?: boolean;
  };
}

export interface DomainConfigList {
  configs: string[];
  examples: string[];
}

export interface DomainConfigInfo {
  name: string;
  path: string;
  config: Record<string, unknown>;
}

export interface DomainConfigCreate {
  name: string;
  system_prompt?: string | null;
  constitution?: string[] | null;
  copy_from?: string | null;
  overwrite?: boolean;
}

export interface ForgeBuildRequest {
  paths: string[];
  task: string;
  base_model: string;
  template?: string;                // "auto" | "alpaca" | "qwen" | ...
  system_prompt: string;
  synth_qa: boolean;
  target_size: number | null;
  output_dir?: string;
  filter_noise?: boolean;
}

/** One of the chat-template registry entries. Kept in sync with
    app/templates/registry.py. The UI renders these as a dropdown. */
export const CHAT_TEMPLATES = [
  "auto", "alpaca", "chatml", "deepseek", "gemma",
  "llama2", "llama3", "mistral", "phi", "qwen", "sharegpt",
] as const;
export type ChatTemplateName = typeof CHAT_TEMPLATES[number];

export interface YouTubeHarvestRequest {
  query: string;
  max_videos: number;
  min_chars?: number;
  output_dir?: string;
}

export interface YouTubeHarvestedFile {
  title: string;
  url: string;
  channel: string;
  language: string;
  auto_generated: boolean;
  char_count: number;
  duration_s: number;
  file_path: string;
}

export interface YouTubeHarvestResponse {
  query: string;
  max_requested: number;
  harvested: YouTubeHarvestedFile[];
  skipped: { title: string; url: string; reason: string }[];
}

export interface ForgeBuildResponse {
  output_path: string;       // .jsonl — training-ready
  preview_path: string;      // _preview.json — first 10 rows pretty-printed
  task: string;
  template: string;
  num_examples: number;
  sources: string[];
}

export interface UploadedFileInfo {
  name: string;
  path: string;
  size: number;
}

export interface UploadResponse {
  uploaded: UploadedFileInfo[];
  skipped: { name: string; reason: string }[];
}

export interface UploadListResponse {
  files: UploadedFileInfo[];
  total_bytes: number;
}

export interface TrainingJobRequest {
  domain_config_name: string;
  base_model: string;
  training_method: string;
  dataset_path?: string | null;
}

export interface LossHistoryEntry {
  step: number;
  loss: number | null;
  learning_rate: number | null;
  grad_norm: number | null;
  epoch: number | null;
  ts: number;
}

export interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  method?: string | null;
  backend?: string | null;
  template?: string | null;
  hardware?: string | null;
  current_loss?: number | null;
  final_loss?: number | null;
  error_message?: string | null;
  adapter_path?: string | null;
  loss_history?: LossHistoryEntry[];
}

export interface ChatRequest {
  message: string;
  domain_config_name: string;
  temperature: number;
  max_new_tokens: number;
}

export interface ChatResponse {
  response: string;
  domain: string;
  model: string;
  backend: string;
  tokens_generated: number;
  ttft_ms: number;
  latency_ms: number;
  tokens_per_second: number;
}

/** A retrieved Docs article (only emitted by the SSE stream in docs mode). */
export interface DocsSource {
  title: string;
  section: string;
  article_id: string;
  score: number;
}
