import { useState } from "react";
import { chat as apiChat, listDomainConfigs } from "../api";
import { useApi } from "../hooks/useApi";
import type { ChatResponse } from "../types";

export default function Chat() {
  const { data: domains } = useApi(listDomainConfigs);

  const [message, setMessage] = useState("");
  const [domain, setDomain] = useState("base");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [sending, setSending] = useState(false);

  const [history, setHistory] = useState<{ role: "user" | "assistant"; text: string; meta?: ChatResponse }[]>([]);

  const handleSend = async () => {
    if (!message.trim()) return;
    const userMsg = message.trim();
    setMessage("");
    setHistory((h) => [...h, { role: "user", text: userMsg }]);
    setSending(true);
    try {
      const res = await apiChat({
        message: userMsg,
        domain_config_name: domain,
        temperature,
        max_new_tokens: maxTokens,
      });
      setHistory((h) => [...h, { role: "assistant", text: res.response, meta: res }]);
    } catch (e) {
      setHistory((h) => [...h, { role: "assistant", text: `Error: ${e}` }]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Chat</h2>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 items-end">
        <label className="text-sm">
          <span className="text-gray-600 font-medium block">Domain</span>
          <select
            className="mt-1 rounded border px-3 py-2 text-sm"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
          >
            <option value="base">base (raw model)</option>
            {(domains?.configs ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label className="text-sm">
          <span className="text-gray-600 font-medium block">Temperature</span>
          <input
            type="number"
            step={0.05}
            min={0}
            max={2}
            className="mt-1 w-20 rounded border px-2 py-2 text-sm"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
        </label>
        <label className="text-sm">
          <span className="text-gray-600 font-medium block">Max tokens</span>
          <input
            type="number"
            step={64}
            min={16}
            max={4096}
            className="mt-1 w-24 rounded border px-2 py-2 text-sm"
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
          />
        </label>
      </div>

      {/* Chat history */}
      <div className="bg-white border rounded-lg p-4 min-h-[300px] max-h-[500px] overflow-y-auto space-y-3">
        {history.length === 0 && (
          <p className="text-gray-400 italic text-sm">No messages yet. Type below to start.</p>
        )}
        {history.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[75%] rounded-lg px-4 py-2 text-sm whitespace-pre-wrap ${
              m.role === "user"
                ? "bg-brand-600 text-white"
                : "bg-gray-100 text-gray-900"
            }`}>
              {m.text}
              {m.meta && (
                <div className="mt-2 text-[10px] opacity-60 font-mono">
                  {m.meta.backend} | {m.meta.ttft_ms.toFixed(0)}ms TTFT |{" "}
                  {m.meta.tokens_generated} tok ({m.meta.tokens_per_second.toFixed(1)} tok/s)
                </div>
              )}
            </div>
          </div>
        ))}
        {sending && <p className="animate-pulse text-gray-400 text-sm">Generating...</p>}
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <textarea
          className="flex-1 rounded border px-3 py-2 text-sm"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type a message..."
          rows={2}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
          }}
        />
        <button onClick={handleSend} disabled={sending} className="btn-primary self-end">
          Send
        </button>
      </div>
    </div>
  );
}
