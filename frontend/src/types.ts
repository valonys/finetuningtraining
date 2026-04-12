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
  system_prompt: string;
  synth_qa: boolean;
  target_size: number | null;
  output_dir?: string;
}

export interface ForgeBuildResponse {
  output_path: string;
  task: string;
  template: string;
  num_examples: number;
  sources: string[];
}

export interface TrainingJobRequest {
  domain_config_name: string;
  base_model: string;
  training_method: string;
  dataset_path?: string | null;
}

export interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  method?: string | null;
  backend?: string | null;
  template?: string | null;
  hardware?: string | null;
  final_loss?: number | null;
  error_message?: string | null;
  adapter_path?: string | null;
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
