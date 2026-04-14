import { useState } from "react";
import { createTrainingJob, getJob, listDomainConfigs } from "../api";
import { useApi } from "../hooks/useApi";

const METHODS = ["sft", "dpo", "orpo", "kto", "grpo"] as const;

export default function Train() {
  const { data: domains } = useApi(listDomainConfigs);

  const [domain, setDomain] = useState("");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-7B-Instruct");
  const [method, setMethod] = useState<string>("sft");
  const [datasetPath, setDatasetPath] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState<string | null>(null);

  // Polling
  const [jobId, setJobId] = useState("");
  const [jobStatus, setJobStatus] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!domain) { setSubmitResult("Pick a domain (create one in the Domains tab first)."); return; }
    if (!datasetPath) { setSubmitResult("Enter a dataset path."); return; }
    setSubmitting(true);
    setSubmitResult(null);
    try {
      const res = await createTrainingJob({
        domain_config_name: domain,
        base_model: baseModel,
        training_method: method,
        dataset_path: datasetPath,
      });
      setSubmitResult(`Job queued: ${res.job_id}`);
      setJobId(res.job_id);
    } catch (e) {
      setSubmitResult(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const handlePoll = async () => {
    if (!jobId.trim()) return;
    try {
      const j = await getJob(jobId.trim());
      const lines = [
        `Status: ${j.status} (${(j.progress * 100).toFixed(0)}%)`,
        `Method: ${j.method}`,
        j.backend ? `Backend: ${j.backend}` : null,
        j.template ? `Template: ${j.template}` : null,
        j.hardware ? `Hardware: ${j.hardware}` : null,
        j.final_loss != null ? `Final loss: ${j.final_loss.toFixed(4)}` : null,
        j.adapter_path ? `Adapter: ${j.adapter_path}` : null,
        j.error_message ? `Error: ${j.error_message}` : null,
      ].filter(Boolean);
      setJobStatus(lines.join("\n"));
    } catch (e) {
      setJobStatus(String(e));
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Train</h2>
      <p className="text-sm text-gray-500">
        Pick a domain (create one in the <b>Domains</b> tab first), a base model,
        a method, and a dataset. Template, hardware backend, and LoRA rank are
        auto-resolved.
      </p>

      <div className="bg-white border rounded-lg p-6 space-y-4">
        <div className="grid md:grid-cols-2 gap-4">
          <label className="block text-sm">
            <span className="text-gray-600 font-medium">Domain</span>
            <select
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
            >
              <option value="">-- select --</option>
              {(domains?.configs ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <label className="block text-sm">
            <span className="text-gray-600 font-medium">Base model</span>
            <input
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
            />
          </label>

          <label className="block text-sm">
            <span className="text-gray-600 font-medium">Method</span>
            <select
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={method}
              onChange={(e) => setMethod(e.target.value)}
            >
              {METHODS.map((m) => <option key={m} value={m}>{m.toUpperCase()}</option>)}
            </select>
          </label>

          <label className="block text-sm">
            <span className="text-gray-600 font-medium">Dataset path</span>
            <input
              className="mt-1 block w-full rounded border px-3 py-2 text-sm font-mono"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              placeholder="./data/processed/my_domain_sft"
            />
          </label>
        </div>

        <button onClick={handleSubmit} disabled={submitting} className="btn-primary">
          {submitting ? "Queueing..." : "Queue training job"}
        </button>

        {submitResult && <p className="text-sm font-mono">{submitResult}</p>}
      </div>

      {/* Job poller */}
      <div className="bg-white border rounded-lg p-6 space-y-4">
        <h3 className="font-semibold">Job Status</h3>
        <div className="flex gap-2">
          <input
            className="flex-1 rounded border px-3 py-2 text-sm font-mono"
            placeholder="Job ID"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
          />
          <button onClick={handlePoll} className="btn-secondary">Poll</button>
        </div>
        {jobStatus && (
          <pre className="bg-gray-50 border rounded p-3 text-sm font-mono whitespace-pre-wrap">{jobStatus}</pre>
        )}
      </div>
    </div>
  );
}
