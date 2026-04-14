import { useState } from "react";
import { createDomainConfig, getDomainConfig, listDomainConfigs } from "../api";
import { useApi } from "../hooks/useApi";

export default function Domains() {
  const { data: list, refresh } = useApi(listDomainConfigs);
  const [creating, setCreating] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  // Form state
  const [name, setName] = useState("");
  const [system, setSystem] = useState("");
  const [rules, setRules] = useState("");
  const [copyFrom, setCopyFrom] = useState("");
  const [overwrite, setOverwrite] = useState(false);

  const handleCreate = async () => {
    if (!name.trim()) { setResult("Please enter a name."); return; }
    setCreating(true);
    setResult(null);
    try {
      const constitution = rules.split("\n").map(r => r.trim()).filter(Boolean);
      const res = await createDomainConfig({
        name: name.trim(),
        system_prompt: system || null,
        constitution: constitution.length ? constitution : null,
        copy_from: copyFrom || null,
        overwrite,
      });
      setResult(`Created ${res.name} at ${res.path}`);
      refresh();
    } catch (e) {
      setResult(String(e));
    } finally {
      setCreating(false);
    }
  };

  const handlePreview = async (n: string) => {
    try {
      const info = await getDomainConfig(n);
      setPreview(JSON.stringify(info.config, null, 2));
    } catch (e) {
      setPreview(String(e));
    }
  };

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-bold">Domain Configs</h2>
      <p className="text-sm text-gray-500">
        One domain per engagement &mdash; <code>asset_integrity</code>,{" "}
        <code>customer_grasps</code>, <code>ai_llm</code>, whatever fits. Each
        produces its own trained adapter at <code>outputs/&lt;name&gt;/</code>.
      </p>

      {/* List */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-white border rounded-lg p-4">
          <h3 className="font-semibold text-sm uppercase tracking-wide text-gray-700 mb-2">
            Your domains
          </h3>
          {list?.configs.length ? (
            <ul className="space-y-1">
              {list.configs.map((c) => (
                <li key={c} className="flex items-center justify-between text-sm">
                  <span className="font-mono">{c}</span>
                  <button onClick={() => handlePreview(c)} className="text-brand-600 hover:underline text-xs">
                    Preview
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-400 italic">None yet &mdash; create one below.</p>
          )}
        </div>

        <div className="bg-white border rounded-lg p-4">
          <h3 className="font-semibold text-sm uppercase tracking-wide text-gray-700 mb-2">
            Seed examples
          </h3>
          <ul className="space-y-1">
            {(list?.examples ?? []).map((e) => (
              <li key={e} className="text-sm font-mono text-gray-600">{e}</li>
            ))}
          </ul>
          <p className="text-xs text-gray-400 mt-2">
            Copy one by setting &quot;Copy from&quot; below.
          </p>
        </div>
      </div>

      {/* Preview */}
      {preview && (
        <pre className="bg-gray-900 text-green-400 text-xs rounded-lg p-4 overflow-x-auto max-h-72 whitespace-pre-wrap">
          {preview}
        </pre>
      )}

      {/* Create form */}
      <div className="bg-white border rounded-lg p-6 space-y-4">
        <h3 className="font-semibold">Create new domain</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <Input label="Name" placeholder="e.g. asset_integrity" value={name} onChange={setName} />
          <Input label="Copy from (optional)" placeholder="e.g. asset_integrity" value={copyFrom} onChange={setCopyFrom} />
        </div>
        <TextArea
          label="System prompt"
          placeholder="You are a senior <role> specialising in <area>..."
          value={system}
          onChange={setSystem}
        />
        <TextArea
          label="Constitution (one rule per line)"
          placeholder={"Always cite the relevant standard.\nNever speculate without noting uncertainty."}
          value={rules}
          onChange={setRules}
          rows={3}
        />
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
          Overwrite if exists
        </label>
        <button onClick={handleCreate} disabled={creating} className="btn-primary">
          {creating ? "Creating..." : "Create domain"}
        </button>
        {result && <p className="text-sm mt-2 font-mono">{result}</p>}
      </div>
    </div>
  );
}

function Input({ label, value, onChange, placeholder }: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string;
}) {
  return (
    <label className="block text-sm">
      <span className="text-gray-600 font-medium">{label}</span>
      <input
        className="mt-1 block w-full rounded border-gray-300 border px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </label>
  );
}

function TextArea({ label, value, onChange, placeholder, rows = 4 }: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string; rows?: number;
}) {
  return (
    <label className="block text-sm">
      <span className="text-gray-600 font-medium">{label}</span>
      <textarea
        className="mt-1 block w-full rounded border-gray-300 border px-3 py-2 text-sm focus:border-brand-500 focus:ring-brand-500"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
      />
    </label>
  );
}
