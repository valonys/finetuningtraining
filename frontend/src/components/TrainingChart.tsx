/**
 * TrainingChart — live SVG loss curve, Unsloth-Studio-style.
 *
 * Zero chart-lib dependency: we draw the polyline ourselves so the app
 * keeps its single-digit MB bundle size. The data shape matches what
 * the backend's LossHistoryCallback appends to JobStatus.loss_history:
 *
 *     { step, loss, learning_rate, grad_norm, epoch, ts }
 *
 * Rendered layers (bottom -> top):
 *   1. Grid + y-axis labels
 *   2. Raw per-step loss polyline (thin, 30% opacity)
 *   3. EMA-smoothed loss polyline (thick, full opacity)
 *   4. Summary chips (current, min, avg, steps) above the chart
 *
 * The parent is responsible for polling /v1/jobs/{id} and feeding us a
 * fresh `history` prop. We recompute the path strings on every render —
 * cheap even for 10k points (our FIFO cap in the backend).
 */
import { useMemo } from "react";
import type { LossHistoryEntry } from "../types";

interface Props {
  history: LossHistoryEntry[];
  status: string;             // "queued" | "training" | "completed" | "failed"
  height?: number;            // SVG viewport height (default 220)
  smoothing?: number;         // EMA alpha in [0,1]. Lower = smoother. Default 0.15.
}

export default function TrainingChart({
  history,
  status,
  height = 220,
  smoothing = 0.15,
}: Props) {
  const series = useMemo(
    () => history.filter((h) => typeof h.loss === "number").map((h) => ({ step: h.step, loss: h.loss as number })),
    [history]
  );

  // EMA smoothing — same formula Unsloth and TensorBoard use.
  const smoothed = useMemo(() => {
    const out: number[] = [];
    let ema = 0;
    series.forEach((p, i) => {
      ema = i === 0 ? p.loss : smoothing * p.loss + (1 - smoothing) * ema;
      out.push(ema);
    });
    return out;
  }, [series, smoothing]);

  if (series.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border bg-gray-50 text-sm text-gray-400"
        style={{ height }}
      >
        {status === "training"
          ? "Waiting for the first logged step..."
          : status === "queued"
          ? "Job is queued — training will start shortly."
          : "No loss history yet."}
      </div>
    );
  }

  // ── Chart viewport & scales ────────────────────────────────
  const W = 720;           // internal viewBox width; SVG scales responsively
  const H = height;
  const padL = 44, padR = 12, padT = 16, padB = 28;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const losses = series.map((p) => p.loss);
  const minL = Math.min(...losses, ...smoothed);
  const maxL = Math.max(...losses, ...smoothed);
  const minStep = series[0].step;
  const maxStep = series[series.length - 1].step;

  // If the run only has one logged step, give x-axis a bit of width so
  // the polyline renders as a point.
  const stepSpan = Math.max(1, maxStep - minStep);
  // Pad the y-axis 5% on each side so the curve doesn't kiss the borders.
  const lossPad = (maxL - minL) * 0.05 || 0.01;
  const yLo = minL - lossPad;
  const yHi = maxL + lossPad;

  const xOf = (step: number) => padL + ((step - minStep) / stepSpan) * innerW;
  const yOf = (v: number) => padT + (1 - (v - yLo) / (yHi - yLo)) * innerH;

  const rawPath = series.map((p, i) => `${i === 0 ? "M" : "L"}${xOf(p.step).toFixed(1)},${yOf(p.loss).toFixed(1)}`).join(" ");
  const smoothPath = series.map((p, i) => `${i === 0 ? "M" : "L"}${xOf(p.step).toFixed(1)},${yOf(smoothed[i]).toFixed(1)}`).join(" ");

  // Y-axis ticks — 4 evenly-spaced values rounded to 3 decimals.
  const ticks = [0, 0.33, 0.66, 1.0].map((t) => yLo + t * (yHi - yLo));

  // Summary chips
  const current = series[series.length - 1].loss;
  const min = Math.min(...losses);
  const avg = losses.reduce((a, b) => a + b, 0) / losses.length;
  const steps = maxStep;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Chip label="current"  value={current.toFixed(4)} tone="emerald" />
        <Chip label="min"      value={min.toFixed(4)}     tone="sky" />
        <Chip label="avg"      value={avg.toFixed(4)}     tone="gray" />
        <Chip label="steps"    value={String(steps)}      tone="gray" />
        {status === "training" && (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-700 ring-1 ring-amber-200">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
            live
          </span>
        )}
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full rounded-lg border bg-white"
        preserveAspectRatio="none"
      >
        {/* grid */}
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={padL} x2={W - padR} y1={yOf(t)} y2={yOf(t)} stroke="#f1f5f9" strokeWidth={1} />
            <text x={padL - 6} y={yOf(t) + 3} textAnchor="end" fontSize={10} fill="#64748b" fontFamily="monospace">
              {t.toFixed(3)}
            </text>
          </g>
        ))}

        {/* x-axis endpoint labels */}
        <text x={padL} y={H - 8} fontSize={10} fill="#64748b" fontFamily="monospace">step {minStep}</text>
        <text x={W - padR} y={H - 8} textAnchor="end" fontSize={10} fill="#64748b" fontFamily="monospace">step {maxStep}</text>

        {/* raw curve (faded) */}
        <path d={rawPath} fill="none" stroke="#93c5fd" strokeWidth={1} opacity={0.55} />
        {/* smoothed curve */}
        <path d={smoothPath} fill="none" stroke="#2563eb" strokeWidth={2} />
        {/* end-point marker */}
        <circle cx={xOf(maxStep)} cy={yOf(current)} r={3.5} fill="#2563eb" />
      </svg>

      <p className="text-xs text-gray-500">
        Blue line: EMA-smoothed loss (alpha = {smoothing}). Faded blue: raw per-log-step loss.
      </p>
    </div>
  );
}

function Chip({ label, value, tone }: { label: string; value: string; tone: "emerald" | "sky" | "gray" }) {
  const cls =
    tone === "emerald" ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
    : tone === "sky"   ? "bg-sky-50 text-sky-700 ring-sky-200"
    :                    "bg-gray-50 text-gray-700 ring-gray-200";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${cls}`}>
      <span className="opacity-60">{label}</span>
      <span className="font-mono">{value}</span>
    </span>
  );
}
