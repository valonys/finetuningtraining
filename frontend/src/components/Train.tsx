import { useEffect, useRef, useState } from "react";
import { createTrainingJob, getJob, listDomainConfigs } from "../api";
import { useApi } from "../hooks/useApi";
import type { JobStatus } from "../types";
import TrainingChart from "./TrainingChart";

const METHODS = ["sft", "dpo", "orpo", "kto", "grpo"] as const;

export default function Train() {
  const { data: domains } = useApi(listDomainConfigs);

  const [domain, setDomain] = useState("");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-7B-Instruct");
  const [method, setMethod] = useState<string>("sft");
  const [datasetPath, setDatasetPath] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitResult, setSubmitResult] = useState<string | null>(null);

  // Job-status poller.
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<JobStatus | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const pollTimer = useRef<number | null>(null);

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
      setJob(res);                     // start the chart immediately
    } catch (e) {
      setSubmitResult(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  // Auto-poll every 2s while a job is active. Stops automatically on
  // completed/failed so we don't hammer the backend after the run ends.
  useEffect(() => {
    const id = jobId.trim();
    if (!id) return;

    let cancelled = false;
    const tick = async () => {
      try {
        const j = await getJob(id);
        if (cancelled) return;
        setJob(j);
        setPollError(null);
        if (j.status === "completed" || j.status === "failed") {
          if (pollTimer.current) {
            clearInterval(pollTimer.current);
            pollTimer.current = null;
          }
        }
      } catch (e) {
        if (!cancelled) setPollError(String(e));
      }
    };
    tick();                              // immediate fetch
    pollTimer.current = window.setInterval(tick, 2000);
    return () => {
      cancelled = true;
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [jobId]);

  const statusBadge = job?.status
    ? <StatusBadge status={job.status} />
    : null;

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

      {/* Training card — live metrics. Follows Unsloth Studio's layout:
          header with status + progress, SVG chart, then metadata chips. */}
      <div className="bg-white border rounded-lg p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">Training</h3>
          <div className="flex items-center gap-3">
            {statusBadge}
            {job?.progress != null && (
              <span className="text-xs text-gray-500 font-mono">
                {(job.progress * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>

        <div className="flex gap-2">
          <input
            className="flex-1 rounded border px-3 py-2 text-sm font-mono"
            placeholder="Job ID (auto-filled after you queue a job)"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
          />
        </div>

        {job?.progress != null && (
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${Math.max(2, job.progress * 100)}%` }}
            />
          </div>
        )}

        <TrainingChart
          history={job?.loss_history ?? []}
          status={job?.status ?? "idle"}
        />

        {/* Metadata grid — only rendered once the backend has populated
            the fields, which happens right after the trainer starts. */}
        {job && (job.backend || job.template || job.hardware || job.final_loss != null) && (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs md:grid-cols-4">
            {job.method && <Meta k="Method" v={job.method.toUpperCase()} />}
            {job.backend && <Meta k="Backend" v={job.backend} />}
            {job.template && <Meta k="Template" v={job.template} />}
            {job.hardware && <Meta k="Hardware" v={job.hardware} />}
            {job.final_loss != null && <Meta k="Final loss" v={job.final_loss.toFixed(4)} />}
            {job.adapter_path && <Meta k="Adapter" v={job.adapter_path} mono />}
          </dl>
        )}

        {pollError && <p className="text-xs text-red-600 font-mono">{pollError}</p>}
        {job?.error_message && (
          <p className="rounded bg-red-50 p-3 text-xs text-red-700 font-mono whitespace-pre-wrap">
            {job.error_message}
          </p>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "training"  ? "bg-amber-50 text-amber-700 ring-amber-200"
    : status === "completed" ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
    : status === "failed"  ? "bg-red-50 text-red-700 ring-red-200"
    :                        "bg-gray-50 text-gray-600 ring-gray-200";
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${cls}`}>
      {status}
    </span>
  );
}

function Meta({ k, v, mono = false }: { k: string; v: string; mono?: boolean }) {
  return (
    <div>
      <dt className="text-gray-500">{k}</dt>
      <dd className={mono ? "font-mono text-gray-800 break-all" : "text-gray-800"}>{v}</dd>
    </div>
  );
}
