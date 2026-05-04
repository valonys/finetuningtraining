import { useState } from "react";
import {
  forgeListUploads,
  multimodalIndex,
  multimodalRag,
  multimodalSearch,
} from "../api";
import { useApi } from "../hooks/useApi";
import type {
  MultimodalIndexResponse,
  MultimodalRAGResponse,
  MultimodalSearchResponse,
  MultimodalSourceType,
} from "../types";

const SOURCE_TYPES: Array<{ value: "" | MultimodalSourceType; label: string }> = [
  { value: "", label: "Auto-detect" },
  { value: "audio", label: "Audio transcript" },
  { value: "slide", label: "Slide / deck" },
  { value: "image", label: "Image" },
  { value: "video", label: "Video description" },
  { value: "document", label: "Document" },
  { value: "text", label: "Text" },
  { value: "code", label: "Code" },
];

export default function Multimodal() {
  const { data: uploads, loading, error, refresh } = useApi(forgeListUploads);
  const [selected, setSelected] = useState<string[]>([]);
  const [collection, setCollection] = useState("default");
  const [sourceType, setSourceType] = useState<"" | MultimodalSourceType>("");
  const [query, setQuery] = useState("budget and product features");
  const [topK, setTopK] = useState(6);
  const [generate, setGenerate] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [indexResult, setIndexResult] = useState<MultimodalIndexResponse | null>(null);
  const [searchResult, setSearchResult] = useState<MultimodalSearchResponse | null>(null);
  const [ragResult, setRagResult] = useState<MultimodalRAGResponse | null>(null);

  const togglePath = (path: string) => {
    setSelected((cur) =>
      cur.includes(path) ? cur.filter((p) => p !== path) : [...cur, path]
    );
  };

  const handleIndex = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const result = await multimodalIndex({
        paths: selected,
        collection,
        source_type: sourceType || null,
        chunk_target_chars: 1200,
        chunk_overlap_chars: 160,
        embedding_dim: 384,
        embed_provider: "hash",
      });
      setIndexResult(result);
      setMessage(`Indexed ${result.chunks_indexed} chunks into ${result.collection}.`);
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleSearch = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const result = await multimodalSearch({
        query,
        collection,
        top_k: topK,
        source_type: sourceType || null,
        embedding_dim: 384,
        embed_provider: "hash",
      });
      setSearchResult(result);
      setRagResult(null);
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRag = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const result = await multimodalRag({
        query,
        collection,
        top_k: topK,
        source_type: sourceType || null,
        embedding_dim: 384,
        embed_provider: "hash",
        generate,
        max_context_chars: 12000,
      });
      setRagResult(result);
      setSearchResult(null);
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  };

  const results = ragResult?.results ?? searchResult?.results ?? [];

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border bg-white shadow-sm overflow-hidden">
        <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-cyan-900 px-6 py-5 text-white">
          <h2 className="text-xl font-semibold">Multimodal Pipeline</h2>
          <p className="text-sm text-slate-300 mt-1">
            Index audio transcripts, OCR text, slides, video descriptions, documents, and code into one cited retrieval layer.
          </p>
        </div>
        <div className="p-6 grid gap-4 md:grid-cols-4">
          <label className="block md:col-span-2">
            <span className="text-sm font-medium text-gray-700">Collection</span>
            <input
              value={collection}
              onChange={(e) => setCollection(e.target.value)}
              className="mt-1 w-full rounded-lg border px-3 py-2 text-sm"
              placeholder="customer-meeting-q2"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium text-gray-700">Modality</span>
            <select
              value={sourceType}
              onChange={(e) => setSourceType(e.target.value as "" | MultimodalSourceType)}
              className="mt-1 w-full rounded-lg border px-3 py-2 text-sm"
            >
              {SOURCE_TYPES.map((s) => (
                <option key={s.value || "auto"} value={s.value}>{s.label}</option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="text-sm font-medium text-gray-700">Top K</span>
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="mt-1 w-full rounded-lg border px-3 py-2 text-sm"
            />
          </label>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <h3 className="font-semibold text-gray-900">1. Select Uploaded Files</h3>
            <button className="text-xs text-blue-600 hover:underline" onClick={refresh}>
              Refresh uploads
            </button>
          </div>
          {loading && <p className="mt-3 text-sm text-gray-500">Loading uploads...</p>}
          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
          <div className="mt-4 max-h-64 overflow-auto divide-y rounded-xl border">
            {(uploads?.files ?? []).map((file) => (
              <label key={file.path} className="flex items-center gap-3 px-3 py-2 text-sm hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={selected.includes(file.path)}
                  onChange={() => togglePath(file.path)}
                />
                <span className="flex-1 truncate" title={file.path}>{file.name}</span>
                <span className="text-xs text-gray-400">{Math.ceil(file.size / 1024)} KB</span>
              </label>
            ))}
            {uploads?.files.length === 0 && (
              <p className="px-3 py-4 text-sm text-gray-500">
                No uploads yet. Add files in Data Forge first.
              </p>
            )}
          </div>
          <button
            onClick={handleIndex}
            disabled={busy || selected.length === 0 || !collection.trim()}
            className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            Index Selected Files
          </button>
          {indexResult && (
            <div className="mt-4 rounded-xl bg-slate-50 p-3 text-xs text-slate-700">
              <div>Records: {indexResult.records_indexed}</div>
              <div>Chunks: {indexResult.chunks_indexed}</div>
              <div>By modality: {JSON.stringify(indexResult.stats.by_modality)}</div>
            </div>
          )}
        </div>

        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <h3 className="font-semibold text-gray-900">2. Query Collection</h3>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={4}
            className="mt-4 w-full rounded-xl border px-3 py-2 text-sm"
            placeholder="Ask across audio, slides, video, and documents..."
          />
          <label className="mt-3 flex items-center gap-2 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={generate}
              onChange={(e) => setGenerate(e.target.checked)}
            />
            Generate final answer with inference backend
          </label>
          <div className="mt-4 flex gap-3">
            <button
              onClick={handleSearch}
              disabled={busy || !query.trim()}
              className="rounded-lg border px-4 py-2 text-sm font-medium disabled:opacity-40"
            >
              Semantic Search
            </button>
            <button
              onClick={handleRag}
              disabled={busy || !query.trim()}
              className="rounded-lg bg-cyan-700 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
            >
              Build RAG Context
            </button>
          </div>
          {message && <p className="mt-3 text-sm text-gray-600">{message}</p>}
        </div>
      </section>

      {(searchResult || ragResult) && (
        <section className="rounded-2xl border bg-white p-5 shadow-sm">
          <h3 className="font-semibold text-gray-900">Results</h3>
          {ragResult && (
            <div className="mt-4 rounded-xl bg-cyan-50 p-4 text-sm text-cyan-950">
              <div className="font-medium">Answer</div>
              <p className="mt-1 whitespace-pre-wrap">{ragResult.answer}</p>
              {ragResult.sources.length > 0 && (
                <p className="mt-3 text-xs text-cyan-700">
                  Sources: {ragResult.sources.join(", ")}
                </p>
              )}
            </div>
          )}
          <div className="mt-4 space-y-3">
            {results.map((r, idx) => (
              <article key={r.chunk_id} className="rounded-xl border p-4">
                <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 uppercase tracking-wide">
                    {r.source_type}
                  </span>
                  <span>#{idx + 1}</span>
                  <span>score {r.score.toFixed(4)}</span>
                  <span className="truncate" title={r.source_uri}>{r.title ?? r.source_uri}</span>
                </div>
                <p className="mt-2 text-sm text-gray-800">{r.text}</p>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
