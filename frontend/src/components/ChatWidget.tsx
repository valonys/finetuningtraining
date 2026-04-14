import { useCallback, useEffect, useRef, useState } from "react";
import { chatStream, listDomainConfigs } from "../api";
import { useApi } from "../hooks/useApi";
import type { ChatResponse } from "../types";

/* ── Types ──────────────────────────────────────────────────── */
interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  meta?: ChatResponse;
  ts: number;
}

/* ── Icons (inline SVG, no deps) ────────────────────────────── */
const ChatIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
    <path d="M4.913 2.658c2.075-.27 4.19-.408 6.337-.408 2.147 0 4.262.139 6.337.408 1.922.25 3.291 1.861 3.405 3.727a4.403 4.403 0 00-1.032-.211 50.89 50.89 0 00-8.42 0c-2.358.196-4.04 2.19-4.04 4.434v4.286a4.47 4.47 0 002.433 3.984L7.28 21.53A.75.75 0 016 21v-4.03a48.527 48.527 0 01-1.087-.128C2.905 16.58 1.5 14.833 1.5 12.862V6.638c0-1.97 1.405-3.718 3.413-3.979z" />
    <path d="M15.75 7.5c-1.376 0-2.739.057-4.086.169C10.124 7.797 9 9.103 9 10.609v4.285c0 1.507 1.128 2.814 2.67 2.94 1.243.102 2.5.157 3.768.165l2.782 2.781a.75.75 0 001.28-.53v-2.39l.33-.026c1.542-.125 2.67-1.433 2.67-2.94v-4.286c0-1.505-1.125-2.811-2.664-2.94A49.392 49.392 0 0015.75 7.5z" />
  </svg>
);

const CloseIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
    <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
  </svg>
);

const MinIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
    <path fillRule="evenodd" d="M14.77 12.79a.75.75 0 01-1.06-.02L10 8.832 6.29 12.77a.75.75 0 11-1.08-1.04l4.25-4.5a.75.75 0 011.08 0l4.25 4.5a.75.75 0 01-.02 1.06z" clipRule="evenodd" />
  </svg>
);

const GearIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
    <path fillRule="evenodd" d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
  </svg>
);

const SendIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
    <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086l-1.414 4.926a.75.75 0 00.826.95 28.897 28.897 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
  </svg>
);

/* ── Typing indicator ───────────────────────────────────────── */
function TypingDots() {
  return (
    <div className="flex items-center gap-1 px-4 py-3">
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  );
}

/* ── Message bubble ─────────────────────────────────────────── */
function Bubble({ msg, streaming = false }: { msg: Message; streaming?: boolean }) {
  const isUser = msg.role === "user";
  // Show a blinking caret after the text while this bubble is actively
  // being streamed (delta-by-delta). Disappears once the stream ends.
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      {!isUser && (
        <div className="flex-shrink-0 mr-2 mt-1">
          <img src="/ValonyLabs_Logo.png" alt="" className="w-7 h-7 rounded-full" />
        </div>
      )}
      <div className={`max-w-[80%] ${isUser ? "order-1" : ""}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-blue-600 text-white rounded-br-md"
              : "bg-gray-100 text-gray-900 rounded-bl-md"
          }`}
        >
          {msg.text}
          {streaming && msg.text && (
            <span className="inline-block w-[0.5em] h-[1em] bg-gray-500 ml-0.5 align-text-bottom animate-pulse" />
          )}
        </div>
        {msg.meta && (
          <div className="mt-1 text-[10px] text-gray-400 font-mono px-1">
            {msg.meta.backend} &middot; {msg.meta.ttft_ms.toFixed(0)}ms TTFT &middot;{" "}
            {msg.meta.tokens_generated} tok ({msg.meta.tokens_per_second.toFixed(1)}/s)
          </div>
        )}
        <div className="text-[10px] text-gray-300 px-1 mt-0.5">
          {new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </div>
      </div>
    </div>
  );
}

/* ── Main widget ────────────────────────────────────────────── */
export default function ChatWidget() {
  const { data: domains } = useApi(listDomainConfigs);

  /* State */
  const [open, setOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [unread, setUnread] = useState(0);

  /* Settings */
  const [domain, setDomain] = useState("base");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  /* Auto-scroll on new messages */
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  /* Focus input when panel opens */
  useEffect(() => {
    if (open) { inputRef.current?.focus(); setUnread(0); }
  }, [open]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");

    // Seed the conversation with the user's message and an empty
    // assistant placeholder. As SSE deltas arrive we append to the
    // placeholder's `text` — React re-renders → typewriter effect.
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text,
      ts: Date.now(),
    };
    const assistantId = crypto.randomUUID();
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      text: "",
      ts: Date.now(),
    };
    setMessages((m) => [...m, userMsg, assistantMsg]);
    setSending(true);

    const updateAssistant = (patch: (m: Message) => Message) => {
      setMessages((all) =>
        all.map((msg) => (msg.id === assistantId ? patch(msg) : msg))
      );
    };

    try {
      await chatStream(
        {
          message: text,
          domain_config_name: domain,
          temperature,
          max_new_tokens: maxTokens,
        },
        {
          onDelta: (delta) => {
            updateAssistant((m) => ({ ...m, text: m.text + delta }));
          },
          onMeta: (meta) => {
            // Merge in the final meta frame (backend, model, TTFT, tokens, ...)
            // so the footnote under the bubble lights up.
            updateAssistant((m) => ({
              ...m,
              meta: { ...(m.meta ?? {}), ...meta } as ChatResponse,
            }));
          },
          onError: (err) => {
            updateAssistant((m) => ({
              ...m,
              text: m.text ? `${m.text}\n\n[stream error: ${err}]` : `Error: ${err}`,
            }));
          },
        }
      );
      if (!open) setUnread((u) => u + 1);
    } catch (e) {
      updateAssistant((m) => ({
        ...m,
        text: m.text || `Error: ${e}`,
      }));
    } finally {
      setSending(false);
    }
  }, [input, sending, domain, temperature, maxTokens, open]);

  return (
    <>
      {/* ── FAB ─────────────────────────────────────────── */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-blue-600
                     text-white shadow-lg hover:bg-blue-700 hover:scale-105
                     transition-all duration-200 flex items-center justify-center
                     focus:outline-none focus:ring-4 focus:ring-blue-300"
          aria-label="Open assistant"
        >
          <ChatIcon />
          {unread > 0 && (
            <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-[10px]
                             font-bold rounded-full flex items-center justify-center">
              {unread}
            </span>
          )}
        </button>
      )}

      {/* ── Panel ───────────────────────────────────────── */}
      {open && (
        <div
          className="fixed bottom-6 right-6 z-50 w-[400px] h-[600px] max-h-[85vh]
                     bg-white rounded-2xl shadow-2xl border border-gray-200
                     flex flex-col overflow-hidden
                     animate-[slideUp_0.25s_ease-out]"
          style={{ maxWidth: "calc(100vw - 2rem)" }}
        >
          {/* Header */}
          <div className="bg-slate-800 text-white px-4 py-3 flex items-center gap-3 flex-shrink-0">
            <img src="/ValonyLabs_Logo.png" alt="" className="w-8 h-8 rounded-full" />
            <div className="flex-1 min-w-0">
              <h3 className="text-sm font-semibold leading-tight">ValonyLabs Assistant</h3>
              <p className="text-[11px] text-slate-400 truncate">
                {domain === "base" ? "Base model" : domain} &middot;{" "}
                {(domains?.configs.length ?? 0) + 1} domains available
              </p>
            </div>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-1.5 rounded-lg hover:bg-slate-700 transition-colors"
              title="Settings"
            >
              <GearIcon />
            </button>
            <button
              onClick={() => setOpen(false)}
              className="p-1.5 rounded-lg hover:bg-slate-700 transition-colors"
              title="Minimize"
            >
              <MinIcon />
            </button>
          </div>

          {/* Settings drawer (slides down inside the panel) */}
          {showSettings && (
            <div className="bg-slate-50 border-b px-4 py-3 space-y-3 flex-shrink-0 animate-[fadeIn_0.15s_ease-out]">
              <label className="block text-xs">
                <span className="font-medium text-gray-600">Domain</span>
                <select
                  className="mt-1 block w-full rounded border px-2 py-1.5 text-xs"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                >
                  <option value="base">base (raw model)</option>
                  {(domains?.configs ?? []).map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </label>
              <div className="flex gap-4">
                <label className="block text-xs flex-1">
                  <span className="font-medium text-gray-600">Temperature</span>
                  <input
                    type="range" min={0} max={2} step={0.05}
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    className="mt-1 w-full accent-blue-600"
                  />
                  <span className="text-gray-400 font-mono">{temperature.toFixed(2)}</span>
                </label>
                <label className="block text-xs w-24">
                  <span className="font-medium text-gray-600">Max tokens</span>
                  <input
                    type="number" min={16} max={4096} step={64}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(Number(e.target.value))}
                    className="mt-1 block w-full rounded border px-2 py-1 text-xs"
                  />
                </label>
              </div>
            </div>
          )}

          {/* Messages */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4">
            {messages.length === 0 && (
              <div className="text-center mt-12">
                <img src="/ValonyLabs_Logo.png" alt="" className="w-16 h-16 mx-auto opacity-20 mb-4" />
                <p className="text-gray-400 text-sm font-medium">
                  How can I help you today?
                </p>
                <p className="text-gray-300 text-xs mt-1">
                  Ask anything about your domain or start a conversation.
                </p>
              </div>
            )}
            {messages.map((m, i) => {
              const isLast = i === messages.length - 1;
              const isStreaming = sending && isLast && m.role === "assistant" && !m.meta;
              return <Bubble key={m.id} msg={m} streaming={isStreaming} />;
            })}
            {(() => {
              // Typing dots only BEFORE the first delta arrives — once the
              // assistant bubble has any text, the growing text itself
              // tells the user something's happening.
              const last = messages[messages.length - 1];
              const preFirstToken =
                sending && last?.role === "assistant" && !last.text;
              return preFirstToken ? <TypingDots /> : null;
            })()}
          </div>

          {/* Input */}
          <div className="border-t bg-white px-3 py-3 flex-shrink-0">
            <div className="flex items-end gap-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
                }}
                placeholder="Type a message..."
                rows={1}
                className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-2.5
                           text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                           outline-none max-h-28 overflow-y-auto"
                style={{ minHeight: "42px" }}
              />
              <button
                onClick={handleSend}
                disabled={sending || !input.trim()}
                className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-600 text-white
                           flex items-center justify-center hover:bg-blue-700
                           disabled:opacity-40 disabled:cursor-not-allowed
                           transition-colors"
                title="Send"
              >
                <SendIcon />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Keyframe animations ─────────────────────────── */}
      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(24px) scale(0.95); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
      `}</style>
    </>
  );
}
