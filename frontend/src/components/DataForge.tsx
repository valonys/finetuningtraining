import { useState } from "react";
import { forgeBuildDataset } from "../api";

const TASKS = ["sft", "dpo", "orpo", "kto", "grpo"] as const;

export default function DataForge() {
  const [paths, setPaths] = useState("");
  const [task, setTask] = useState<string>("sft");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-7B-Instruct");
  const [sysPrompt, setSysPrompt] = useState("You are a helpful assistant.");
  const [synthQa, setSynthQa] = useState(true);
  const [targetSize, setTargetSize] = useState(500);
  const [building, setBuilding] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleBuild = async () => {
    const pathList = paths.split("\n").map((p) => p.trim()).filter(Boolean);
    if (!pathList.length) { setResult("Enter at least one file path."); return; }
    setBuilding(true);
    setResult(null);
    try {
      const res = await forgeBuildDataset({
        paths: pathList,
        task,
        base_model: baseModel,
        system_prompt: sysPrompt,
        synth_qa: synthQa,
        target_size: targetSize || null,
      });
      setResult(
        `Built ${res.task} dataset with template "${res.template}"\n` +
        `${res.num_examples} examples from ${res.sources.length} source(s)\n` +
        `Saved to: ${res.output_path}`
      );
    } catch (e) {
      setResult(String(e));
    } finally {
      setBuilding(false);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Data Forge</h2>
      <p className="text-sm text-gray-500">
        Ingest PDFs, DOCX, XLSX, PPTX, images, HTML, TXT, CSV &mdash; generate
        a training dataset with the correct chat template for your base model.
        Set <code>OLLAMA_API_KEY</code> for Nemotron-powered Q/A synthesis.
      </p>

      <div className="bg-white border rounded-lg p-6 space-y-4">
        <label className="block text-sm">
          <span className="text-gray-600 font-medium">
            File paths (one per line, server-side paths)
          </span>
          <textarea
            className="mt-1 block w-full rounded border px-3 py-2 text-sm font-mono"
            value={paths}
            onChange={(e) => setPaths(e.target.value)}
            placeholder={"./data/uploads/report.pdf\n./data/uploads/spec.docx"}
            rows={4}
          />
        </label>

        <div className="grid md:grid-cols-3 gap-4">
          <label className="block text-sm">
            <span className="text-gray-600 font-medium">Task</span>
            <select
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={task}
              onChange={(e) => setTask(e.target.value)}
            >
              {TASKS.map((t) => <option key={t} value={t}>{t.toUpperCase()}</option>)}
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
            <span className="text-gray-600 font-medium">Target size</span>
            <input
              type="number"
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={targetSize}
              onChange={(e) => setTargetSize(Number(e.target.value))}
            />
          </label>
        </div>

        <label className="block text-sm">
          <span className="text-gray-600 font-medium">System prompt</span>
          <input
            className="mt-1 block w-full rounded border px-3 py-2 text-sm"
            value={sysPrompt}
            onChange={(e) => setSysPrompt(e.target.value)}
          />
        </label>

        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={synthQa} onChange={(e) => setSynthQa(e.target.checked)} />
          Synthesize Q/A pairs (uses the configured synth provider)
        </label>

        <button onClick={handleBuild} disabled={building} className="btn-primary">
          {building ? "Building..." : "Build dataset"}
        </button>

        {result && (
          <pre className="bg-gray-50 border rounded p-3 text-sm font-mono whitespace-pre-wrap">{result}</pre>
        )}
      </div>
    </div>
  );
}
