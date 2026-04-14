import { useState } from "react";
import Health from "./components/Health";
import Domains from "./components/Domains";
import DataForge from "./components/DataForge";
import Train from "./components/Train";
import ChatWidget from "./components/ChatWidget";
import DocsSidebar from "./components/DocsSidebar";

const TABS = [
  { id: "health",    label: "Health" },
  { id: "domains",   label: "Domains" },
  { id: "dataforge", label: "Data Forge" },
  { id: "train",     label: "Train" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<TabId>("health");
  const [docsOpen, setDocsOpen] = useState(false);

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Header ──────────────────────────────────────── */}
      <header className="bg-slate-800 text-white px-6 py-3 flex items-center gap-4 shadow-md">
        <img
          src="/ValonyLabs_Logo.png"
          alt="ValonyLabs"
          className="h-9 w-9 rounded"
        />
        <div className="flex-1">
          <h1 className="text-lg font-semibold leading-tight">ValonyLabs Studio</h1>
          <p className="text-xs text-slate-400">
            Agnostic post-training &amp; inference platform
          </p>
        </div>
        <button
          onClick={() => setDocsOpen(true)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg
                     bg-slate-700 hover:bg-slate-600 text-sm font-medium
                     transition-colors"
          title="Open documentation (guides & tutorials)"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
            <path fillRule="evenodd" d="M11.25 4.533A9.707 9.707 0 006 3a9.735 9.735 0 00-3.25.555.75.75 0 00-.5.707v14.25a.75.75 0 001 .707A8.237 8.237 0 016 18.75c1.995 0 3.823.707 5.25 1.886V4.533zM12.75 20.636A8.214 8.214 0 0118 18.75c.966 0 1.89.166 2.75.47a.75.75 0 001-.708V4.262a.75.75 0 00-.5-.707A9.735 9.735 0 0018 3a9.707 9.707 0 00-5.25 1.533v16.103z" clipRule="evenodd" />
          </svg>
          Docs
        </button>
      </header>

      {/* ── Tab bar ─────────────────────────────────────── */}
      <nav className="bg-white border-b flex gap-1 px-6 pt-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
              tab === t.id
                ? "bg-blue-50 text-blue-700 border-b-2 border-blue-600"
                : "text-gray-500 hover:text-gray-800 hover:bg-gray-50"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* ── Content ─────────────────────────────────────── */}
      <main className="flex-1 p-6 max-w-6xl mx-auto w-full">
        {tab === "health"    && <Health />}
        {tab === "domains"   && <Domains />}
        {tab === "dataforge" && <DataForge />}
        {tab === "train"     && <Train />}
      </main>

      {/* ── Footer ──────────────────────────────────────── */}
      <footer className="text-center text-xs text-gray-400 py-4 border-t">
        ValonyLabs Studio v3.0 &mdash; React + TypeScript + FastAPI
      </footer>

      {/* ── Floating chat widget (always visible, all tabs) */}
      <ChatWidget />

      {/* ── Docs drawer (slides in from the left) */}
      <DocsSidebar open={docsOpen} onClose={() => setDocsOpen(false)} />
    </div>
  );
}
