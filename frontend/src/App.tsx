import { useState } from "react";
import Health from "./components/Health";
import Domains from "./components/Domains";
import DataForge from "./components/DataForge";
import Train from "./components/Train";
import ChatWidget from "./components/ChatWidget";

const TABS = [
  { id: "health",    label: "Health" },
  { id: "domains",   label: "Domains" },
  { id: "dataforge", label: "Data Forge" },
  { id: "train",     label: "Train" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<TabId>("health");

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Header ──────────────────────────────────────── */}
      <header className="bg-slate-800 text-white px-6 py-3 flex items-center gap-4 shadow-md">
        <img
          src="/ValonyLabs_Logo.png"
          alt="ValonyLabs"
          className="h-9 w-9 rounded"
        />
        <div>
          <h1 className="text-lg font-semibold leading-tight">ValonyLabs Studio</h1>
          <p className="text-xs text-slate-400">
            Agnostic post-training &amp; inference platform
          </p>
        </div>
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
    </div>
  );
}
