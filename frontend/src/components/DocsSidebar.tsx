import { useEffect, useMemo, useState } from "react";
import { DOC_SECTIONS, type DocArticle } from "../docs";

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function DocsSidebar({ open, onClose }: Props) {
  const allArticles = useMemo(
    () => DOC_SECTIONS.flatMap((s) => s.articles.map((a) => ({ ...a, sectionId: s.id }))),
    []
  );

  const [activeId, setActiveId] = useState<string>(allArticles[0]?.id ?? "");
  const [expanded, setExpanded] = useState<Record<string, boolean>>(
    () => Object.fromEntries(DOC_SECTIONS.map((s) => [s.id, true]))
  );
  const [query, setQuery] = useState("");

  /* Close on Escape */
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  /* Search filter */
  const filteredSections = useMemo(() => {
    if (!query.trim()) return DOC_SECTIONS;
    const q = query.toLowerCase();
    return DOC_SECTIONS
      .map((s) => ({
        ...s,
        articles: s.articles.filter(
          (a) =>
            a.title.toLowerCase().includes(q) ||
            a.summary.toLowerCase().includes(q)
        ),
      }))
      .filter((s) => s.articles.length > 0);
  }, [query]);

  const activeArticle: (DocArticle & { sectionId: string }) | undefined = allArticles.find(
    (a) => a.id === activeId
  );

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="fixed inset-0 bg-black/30 z-40 animate-[fadeIn_0.15s_ease-out]"
      />

      {/* Drawer */}
      <aside
        className="fixed top-0 left-0 bottom-0 z-50 w-[min(900px,95vw)]
                   bg-white shadow-2xl flex flex-col
                   animate-[slideInLeft_0.25s_ease-out]"
      >
        {/* Header */}
        <div className="bg-slate-800 text-white px-5 py-3 flex items-center gap-3 flex-shrink-0">
          <img src="/ValonyLabs_Logo.png" alt="" className="w-7 h-7 rounded" />
          <div className="flex-1 min-w-0">
            <h2 className="text-sm font-semibold leading-tight">Documentation</h2>
            <p className="text-[11px] text-slate-400">Guides · Tutorials · FAQ</p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-700 transition-colors"
            aria-label="Close docs"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>
        </div>

        {/* Body: two-column nav + content */}
        <div className="flex-1 min-h-0 flex">
          {/* Nav column */}
          <nav className="w-72 border-r bg-gray-50 flex flex-col overflow-hidden">
            <div className="p-3 border-b bg-white">
              <input
                type="search"
                placeholder="Search docs..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm
                           focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <div className="flex-1 overflow-y-auto py-2">
              {filteredSections.map((section) => {
                const isOpen = expanded[section.id] ?? true;
                return (
                  <div key={section.id} className="mb-1">
                    <button
                      onClick={() =>
                        setExpanded((e) => ({ ...e, [section.id]: !isOpen }))
                      }
                      className="w-full flex items-center gap-1 px-3 py-1.5 text-xs
                                 font-semibold uppercase tracking-wide text-gray-500
                                 hover:text-gray-800 hover:bg-gray-100 transition-colors"
                    >
                      <span className={`transition-transform ${isOpen ? "rotate-90" : ""}`}>›</span>
                      {section.title}
                    </button>
                    {isOpen && (
                      <ul className="ml-2">
                        {section.articles.map((a) => {
                          const active = a.id === activeId;
                          return (
                            <li key={a.id}>
                              <button
                                onClick={() => setActiveId(a.id)}
                                className={`w-full text-left pl-6 pr-3 py-1.5 text-sm
                                            border-l-2 transition-colors ${
                                  active
                                    ? "bg-blue-50 text-blue-700 border-blue-600 font-medium"
                                    : "text-gray-700 border-transparent hover:bg-gray-100 hover:text-gray-900"
                                }`}
                              >
                                {a.title}
                              </button>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                );
              })}
              {filteredSections.length === 0 && (
                <p className="px-4 py-3 text-xs text-gray-400 italic">No results for "{query}".</p>
              )}
            </div>
          </nav>

          {/* Content column */}
          <article className="flex-1 overflow-y-auto px-8 py-6">
            {activeArticle ? (
              <>
                <div className="text-xs text-gray-400 font-medium mb-1">
                  {DOC_SECTIONS.find((s) => s.id === activeArticle.sectionId)?.title}
                </div>
                <h1 className="text-2xl font-bold text-gray-900 mb-1">{activeArticle.title}</h1>
                <p className="text-sm text-gray-500 mb-6">{activeArticle.summary}</p>
                {activeArticle.body}
              </>
            ) : (
              <p className="text-gray-400">Select an article from the sidebar.</p>
            )}
          </article>
        </div>
      </aside>

      <style>{`
        @keyframes slideInLeft {
          from { transform: translateX(-100%); }
          to   { transform: translateX(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
      `}</style>
    </>
  );
}
