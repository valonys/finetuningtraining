import { getHealth } from "../api";
import { useApi } from "../hooks/useApi";

export default function Health() {
  const { data, error, loading, refresh } = useApi(getHealth);

  if (loading) return <p className="animate-pulse text-gray-500">Loading health...</p>;
  if (error) return <p className="text-red-600">API unreachable: {error}</p>;
  if (!data) return null;

  const h = data;
  const hw = h.hardware as Record<string, unknown>;
  const p = h.profile as Record<string, unknown>;
  const sp = h.synth_provider;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">System Health</h2>
        <button onClick={refresh} className="btn-secondary">Refresh</button>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <Card title="Hardware">
          <KV label="Status" value={h.status} />
          <KV label="Version" value={h.version} />
          <KV label="Tier" value={String(hw.tier)} />
          <KV label="Device" value={String(hw.device_name)} />
          <KV label="Memory" value={`${hw.effective_memory_gb} GB`} />
          <KV label="bf16" value={hw.supports_bf16 ? "Yes" : "No"} />
          <KV label="FlashAttn" value={hw.supports_flash_attn ? "Yes" : "No"} />
        </Card>

        <Card title="Backends">
          <KV label="Training" value={String(p.training_backend)} />
          <KV label="Inference" value={h.inference_backend} />
          <KV label="torch dtype" value={String(p.torch_dtype)} />
          <KV label="LoRA r" value={String(p.lora_r)} />
          <KV label="Max seq len" value={String(p.max_seq_length)} />
        </Card>

        <Card title="Synth Provider">
          <KV label="Provider" value={sp.provider} />
          <KV label="Model" value={sp.model ?? "(none)"} />
          <KV label="Cloud" value={sp.is_cloud ? "Yes" : "No"} />
          <KV label="URL" value={sp.base_url ?? "(not set)"} />
        </Card>

        <Card title="Capabilities">
          <KV label="Trained adapters" value={h.registered_domains.join(", ") || "(none)"} />
          <KV label="OCR engines" value={h.available_ocr.join(", ")} />
          <KV label="Templates" value={h.available_templates.join(", ")} />
        </Card>
      </div>

      {Object.keys(h.latency_stats).length > 0 && (
        <Card title="Latency (ms)">
          {Object.entries(h.latency_stats).map(([k, v]) => (
            <KV key={k} label={k} value={v.toFixed(1)} />
          ))}
        </Card>
      )}
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-lg border p-4 shadow-sm">
      <h3 className="font-semibold text-gray-700 mb-3 text-sm uppercase tracking-wide">{title}</h3>
      <dl className="space-y-1.5">{children}</dl>
    </div>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm">
      <dt className="text-gray-500">{label}</dt>
      <dd className="font-mono text-gray-900 text-right max-w-[60%] truncate" title={value}>{value}</dd>
    </div>
  );
}
