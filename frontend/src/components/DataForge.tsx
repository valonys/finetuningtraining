import { useCallback, useEffect, useRef, useState } from "react";
import {
  forgeBuildDataset,
  forgeHarvestYoutube,
  forgeClearUploads,
  forgeDeleteUpload,
  forgeListUploads,
  forgeUpload,
} from "../api";
import type { UploadedFileInfo } from "../types";

const TASKS = ["sft", "dpo", "orpo", "kto", "grpo"] as const;

const ACCEPT =
  ".pdf,.txt,.md,.rst,.docx,.pptx,.xlsx,.xls,.csv,.tsv," +
  ".html,.htm,.xhtml,.png,.jpg,.jpeg,.webp,.tiff,.tif,.bmp,.json,.jsonl";

/* ── Icons ─────────────────────────────────────────────────── */
const UploadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-10 h-10">
    <path fillRule="evenodd" d="M12 2.25a.75.75 0 01.75.75v11.69l3.22-3.22a.75.75 0 111.06 1.06l-4.5 4.5a.75.75 0 01-1.06 0l-4.5-4.5a.75.75 0 111.06-1.06l3.22 3.22V3a.75.75 0 01.75-.75zm-9 13.5a.75.75 0 01.75.75v2.25a1.5 1.5 0 001.5 1.5h13.5a1.5 1.5 0 001.5-1.5V16.5a.75.75 0 011.5 0v2.25a3 3 0 01-3 3H5.25a3 3 0 01-3-3V16.5a.75.75 0 01.75-.75z" clipRule="evenodd" />
  </svg>
);

const TrashIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
    <path fillRule="evenodd" d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.52.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4z" clipRule="evenodd" />
  </svg>
);

const FileIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-gray-400">
    <path d="M3 3.5A1.5 1.5 0 014.5 2h6.879a1.5 1.5 0 011.06.44l4.122 4.12A1.5 1.5 0 0117 7.622V16.5a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 013 16.5v-13z" />
  </svg>
);

/* ── Helpers ───────────────────────────────────────────────── */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/* ── Component ─────────────────────────────────────────────── */
export default function DataForge() {
  const [uploaded, setUploaded] = useState<UploadedFileInfo[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<{ loaded: number; total: number } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [skipped, setSkipped] = useState<{ name: string; reason: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* Dataset build form */
  const [task, setTask] = useState<string>("sft");
  const [baseModel, setBaseModel] = useState("Qwen/Qwen2.5-7B-Instruct");
  const [template, setTemplate] = useState<string>("auto");     // chat-template override
  const [filterNoise, setFilterNoise] = useState(true);         // drop TOC / covers / bibliography

  /* YouTube harvester form */
  const [ytQuery, setYtQuery] = useState("");
  const [ytMaxVideos, setYtMaxVideos] = useState(10);
  const [harvesting, setHarvesting] = useState(false);
  const [harvestResult, setHarvestResult] = useState<string | null>(null);
  const [sysPrompt, setSysPrompt] = useState("You are a helpful assistant.");
  const [synthQa, setSynthQa] = useState(true);
  const [targetSize, setTargetSize] = useState(500);
  const [building, setBuilding] = useState(false);
  const [buildResult, setBuildResult] = useState<string | null>(null);

  /* Hydrate uploaded list on mount */
  const refresh = useCallback(async () => {
    try {
      const res = await forgeListUploads();
      setUploaded(res.files);
    } catch (e) {
      // Backend may be offline — ignore; user will see empty state.
      console.error(e);
    }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  /* ── Upload flow ─────────────────────────────────────── */
  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const list = Array.from(files);
    if (list.length === 0) return;
    setUploadError(null);
    setSkipped([]);
    setUploading(true);
    setProgress({ loaded: 0, total: list.reduce((s, f) => s + f.size, 0) });
    try {
      const res = await forgeUpload(list, (loaded, total) => {
        setProgress({ loaded, total });
      });
      setSkipped(res.skipped);
      await refresh();
    } catch (e) {
      setUploadError(String(e));
    } finally {
      setUploading(false);
      setProgress(null);
    }
  }, [refresh]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const onPickClick = () => fileInputRef.current?.click();

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) handleFiles(e.target.files);
    e.target.value = "";   // allow re-selecting the same file
  };

  const handleDelete = async (name: string) => {
    try {
      await forgeDeleteUpload(name);
      await refresh();
    } catch (e) {
      setUploadError(String(e));
    }
  };

  const handleClearAll = async () => {
    if (!confirm(`Delete all ${uploaded.length} uploaded files?`)) return;
    try {
      await forgeClearUploads();
      await refresh();
    } catch (e) {
      setUploadError(String(e));
    }
  };

  /* ── Build flow ──────────────────────────────────────── */
  const handleBuild = async () => {
    if (uploaded.length === 0) {
      setBuildResult("Upload at least one file first.");
      return;
    }
    setBuilding(true);
    setBuildResult(null);
    try {
      const res = await forgeBuildDataset({
        paths: uploaded.map((f) => f.path),
        task,
        base_model: baseModel,
        template: template,              // "auto" or specific family name
        system_prompt: sysPrompt,
        synth_qa: synthQa,
        target_size: targetSize || null,
        filter_noise: filterNoise,
      });
      setBuildResult(
        `Built ${res.task} dataset with template "${res.template}"\n` +
        `${res.num_examples} examples from ${res.sources.length} source(s)\n\n` +
        `Full dataset (JSONL, training-ready):\n  ${res.output_path}\n\n` +
        `Human-readable preview (first 10 rows, pretty JSON):\n  ${res.preview_path}\n\n` +
        `Peek with:\n  jq '.' < ${res.preview_path}\n` +
        `  head -1 ${res.output_path} | jq '.'`
      );
    } catch (e) {
      setBuildResult(String(e));
    } finally {
      setBuilding(false);
    }
  };

  /* ── YouTube harvester flow ─────────────────────────── */
  const handleHarvestYoutube = async () => {
    if (!ytQuery.trim()) {
      setHarvestResult("Enter a keyword query first.");
      return;
    }
    setHarvesting(true);
    setHarvestResult(null);
    try {
      const res = await forgeHarvestYoutube({
        query: ytQuery.trim(),
        max_videos: ytMaxVideos,
      });
      const ok = res.harvested.length;
      const skipped = res.skipped.length;
      const lines = [
        `Harvested ${ok} of ${res.max_requested} videos for query "${res.query}".`,
        "",
        ...res.harvested.map(
          (h, i) =>
            `${String(i + 1).padStart(2, "0")}. ${h.title}\n` +
            `    ${h.channel} | ${h.char_count.toLocaleString()} chars | ` +
            `${h.language}${h.auto_generated ? " (auto)" : ""}\n` +
            `    ${h.file_path}`
        ),
      ];
      if (skipped > 0) {
        lines.push("", `Skipped (${skipped}):`);
        for (const s of res.skipped) {
          lines.push(`  - ${s.title}: ${s.reason}`);
        }
      }
      lines.push("", "Refresh the Uploaded files list below to see the new .txt files.");
      setHarvestResult(lines.join("\n"));
      await refresh();   // re-pull the uploads list so the new files appear
    } catch (e) {
      setHarvestResult(String(e));
    } finally {
      setHarvesting(false);
    }
  };

  const totalBytes = uploaded.reduce((s, f) => s + f.size, 0);
  const pct = progress ? Math.round((progress.loaded / Math.max(1, progress.total)) * 100) : 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold">Data Forge</h2>
        <p className="text-sm text-gray-500 mt-1">
          Drag &amp; drop your raw documents &mdash; PDFs, DOCX, XLSX, PPTX, images, HTML,
          TXT, CSV. The Data Forge routes each format through the right parser + OCR
          pipeline and produces a training dataset with the correct chat template for
          your base model.
        </p>
      </div>

      {/* ── Dropzone ────────────────────────────────── */}
      <div
        onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setDragging(false); }}
        onDrop={onDrop}
        onClick={onPickClick}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer
                    transition-colors ${
          dragging
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 bg-gray-50 hover:bg-gray-100 hover:border-gray-400"
        }`}
      >
        <div className="flex flex-col items-center gap-3 text-gray-500">
          <UploadIcon />
          <div>
            <p className="text-base font-medium text-gray-700">
              {dragging ? "Drop files to upload" : "Drop files here, or click to browse"}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              PDF &middot; DOCX &middot; XLSX &middot; PPTX &middot; HTML &middot; TXT &middot;
              CSV &middot; PNG/JPG/WEBP &middot; JSON/JSONL &mdash; up to 512&nbsp;MB per file
            </p>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ACCEPT}
          onChange={onFileInputChange}
          className="hidden"
        />
      </div>

      {/* Upload progress */}
      {uploading && progress && (
        <div className="bg-white border rounded-lg p-3">
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span>Uploading... {formatBytes(progress.loaded)} / {formatBytes(progress.total)}</span>
            <span>{pct}%</span>
          </div>
          <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all duration-150"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {/* Upload error */}
      {uploadError && (
        <div className="bg-red-50 border-l-4 border-red-400 text-red-800 px-4 py-2 text-sm">
          {uploadError}
        </div>
      )}

      {/* Skipped files */}
      {skipped.length > 0 && (
        <div className="bg-amber-50 border-l-4 border-amber-400 text-amber-900 px-4 py-2 text-sm">
          <div className="font-medium mb-1">Some files were skipped:</div>
          <ul className="list-disc ml-5 space-y-0.5">
            {skipped.map((s, i) => (
              <li key={i}><b>{s.name}</b> &mdash; {s.reason}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ── YouTube harvester ───────────────────────── */}
      <div className="bg-white border rounded-lg p-5 space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-gray-900">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 text-red-600">
            <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" />
          </svg>
          Harvest transcripts from YouTube
        </div>
        <p className="text-xs text-gray-500">
          Keyword search finds the top N matching videos on YouTube and fetches
          their captions (no audio download, no Whisper, no API key). Each
          transcript lands as a <code>.txt</code> file under <code>./data/uploads/</code>{" "}
          and appears in the file list below, ready to be built into a dataset.
        </p>
        <div className="grid md:grid-cols-[1fr_auto] gap-3 items-end">
          <label className="block text-sm">
            <span className="text-gray-600 font-medium">Keyword query</span>
            <input
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={ytQuery}
              onChange={(e) => setYtQuery(e.target.value)}
              placeholder="e.g. asset integrity inspection FPSO"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !harvesting) handleHarvestYoutube();
              }}
            />
          </label>
          <label className="block text-sm">
            <span className="text-gray-600 font-medium">Max videos</span>
            <input
              type="number"
              min={1}
              max={25}
              className="mt-1 block w-24 rounded border px-3 py-2 text-sm"
              value={ytMaxVideos}
              onChange={(e) => setYtMaxVideos(Number(e.target.value))}
            />
          </label>
        </div>
        <button
          onClick={handleHarvestYoutube}
          disabled={harvesting || !ytQuery.trim()}
          className="btn-primary"
        >
          {harvesting ? "Harvesting transcripts..." : "Harvest from YouTube"}
        </button>
        {harvestResult && (
          <pre className="bg-gray-50 border rounded p-3 text-xs font-mono whitespace-pre-wrap max-h-72 overflow-y-auto">
            {harvestResult}
          </pre>
        )}
      </div>

      {/* ── File list ───────────────────────────────── */}
      <div className="bg-white border rounded-lg overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b bg-gray-50">
          <div className="text-sm font-semibold text-gray-700">
            Uploaded files
            {uploaded.length > 0 && (
              <span className="ml-2 text-gray-400 font-normal">
                ({uploaded.length} &middot; {formatBytes(totalBytes)})
              </span>
            )}
          </div>
          <div className="flex gap-3">
            <button onClick={refresh} className="text-xs text-gray-500 hover:text-gray-800">
              Refresh
            </button>
            {uploaded.length > 0 && (
              <button onClick={handleClearAll} className="text-xs text-red-600 hover:text-red-800">
                Clear all
              </button>
            )}
          </div>
        </div>
        {uploaded.length === 0 ? (
          <p className="px-4 py-6 text-sm text-gray-400 italic text-center">
            No files uploaded yet. Drop some above to get started.
          </p>
        ) : (
          <ul className="divide-y">
            {uploaded.map((f) => (
              <li key={f.path} className="px-4 py-2.5 flex items-center gap-3 hover:bg-gray-50">
                <FileIcon />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-900 truncate" title={f.name}>
                    {f.name}
                  </div>
                  <div className="text-xs text-gray-500 font-mono truncate" title={f.path}>
                    {f.path}
                  </div>
                </div>
                <div className="text-xs text-gray-500 font-mono flex-shrink-0 w-20 text-right">
                  {formatBytes(f.size)}
                </div>
                <button
                  onClick={() => handleDelete(f.name)}
                  className="p-1.5 rounded text-gray-400 hover:text-red-600 hover:bg-red-50"
                  title={`Delete ${f.name}`}
                >
                  <TrashIcon />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* ── Build dataset ───────────────────────────── */}
      <div className="bg-white border rounded-lg p-6 space-y-4">
        <h3 className="font-semibold text-gray-900">Build training dataset</h3>

        <div className="grid md:grid-cols-2 gap-4">
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
            <span className="text-gray-600 font-medium">Target size (rows)</span>
            <input
              type="number"
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={targetSize}
              onChange={(e) => setTargetSize(Number(e.target.value))}
            />
          </label>
          <label className="block text-sm">
            <span className="text-gray-600 font-medium">Base model</span>
            <input
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
              placeholder="Qwen/Qwen2.5-7B-Instruct"
            />
          </label>
          <label className="block text-sm">
            <span className="text-gray-600 font-medium">
              Chat template
              {template === "auto" && (
                <span className="ml-2 text-xs text-gray-400 font-normal">
                  (auto-resolved from base model)
                </span>
              )}
            </span>
            <select
              className="mt-1 block w-full rounded border px-3 py-2 text-sm"
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
            >
              <option value="auto">Auto (match base model)</option>
              <optgroup label="Instruction-tuned families">
                <option value="qwen">Qwen (Qwen2 / Qwen2.5 / Qwen3)</option>
                <option value="llama3">Llama 3 / 3.1 / 3.2 / 3.3</option>
                <option value="llama2">Llama 2</option>
                <option value="mistral">Mistral / Mixtral</option>
                <option value="gemma">Gemma 1 / 2 / 3</option>
                <option value="phi">Phi 3 / 4</option>
                <option value="deepseek">DeepSeek V2 / V3 / R1</option>
              </optgroup>
              <optgroup label="Generic formats">
                <option value="chatml">ChatML (OpenAI / Nous / Yi)</option>
                <option value="alpaca">Alpaca (Stanford / Vicuna)</option>
                <option value="sharegpt">ShareGPT</option>
              </optgroup>
            </select>
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

        <div className="flex flex-wrap gap-4">
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={synthQa} onChange={(e) => setSynthQa(e.target.checked)} />
          Synthesize Q/A pairs (uses the configured synth provider)
        </label>
        <label className="flex items-center gap-2 text-sm" title="Reject chunks that look like book front-matter, TOCs, and bibliography fragments before Q/A synthesis.">
          <input
            type="checkbox"
            checked={filterNoise}
            onChange={(e) => setFilterNoise(e.target.checked)}
          />
          Filter noise (drop covers / TOCs / bibliographies)
        </label>
        </div>

        <button
          onClick={handleBuild}
          disabled={building || uploaded.length === 0}
          className="btn-primary"
        >
          {building ? "Building..." : `Build dataset from ${uploaded.length} file${uploaded.length === 1 ? "" : "s"}`}
        </button>

        {buildResult && (
          <pre className="bg-gray-50 border rounded p-3 text-sm font-mono whitespace-pre-wrap">{buildResult}</pre>
        )}
      </div>
    </div>
  );
}
