"use client";

import { FormEvent, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowRight, BadgeCheck, Brain, Loader2, Sparkles, Table2 } from "lucide-react";
import type { CompareResult } from "@/lib/types";
import { defaultPrompt } from "@/lib/default-prompt";

const tiers = [
  "stan-1.5-mini-predictive",
  "stan-1.5-mini-thinking",
  "stan-1.5-mini-instant",
];

export default function Home() {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [systemPrompt, setSystemPrompt] = useState("Use Markdown. Prefer concise tables when useful.");
  const [modelTier, setModelTier] = useState(tiers[0]);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [directTab, setDirectTab] = useState<"answer" | "thinking">("answer");
  const [optimizedTab, setOptimizedTab] = useState<"answer" | "thinking">("answer");

  const savings = useMemo(() => {
    if (!result) return null;
    const before = result.cosavu.totalOriginalTokens || 0;
    const after = result.cosavu.totalOptimizedTokens || 0;
    const saved = Math.max(0, before - after);
    return {
      saved,
      pct: before ? (saved / before) * 100 : 0,
    };
  }, [result]);

  const costSavings = useMemo(() => {
    if (!result) return null;
    const direct = result.raw.costUsd;
    const optimized = result.optimized.costUsd;
    const saved = direct - optimized;
    return {
      direct,
      optimized,
      saved,
      pct: direct ? (saved / direct) * 100 : 0,
    };
  }, [result]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, systemPrompt, modelTier }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Comparison failed");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-4 px-4 py-4 md:px-6">
        <header className="flex min-h-20 items-center justify-between gap-4 rounded-lg border bg-card px-5 py-4">
          <div>
            <p className="eyebrow">Cosavu ContextAPI</p>
            <h1 className="text-2xl font-semibold tracking-normal">Clean Side-by-Side Chat</h1>
          </div>
          <div className="flex items-center gap-2 rounded-md border bg-muted px-3 py-2 text-sm text-muted-foreground">
            <BadgeCheck className="h-4 w-4" />
            Cosavu-only UI
          </div>
        </header>

        <section className="rounded-lg border bg-card p-4">
          <form onSubmit={submit} className="grid gap-3">
            <div className="grid gap-2">
              <label htmlFor="systemPrompt" className="label">System instruction</label>
              <textarea
                id="systemPrompt"
                className="textarea min-h-16"
                value={systemPrompt}
                onChange={(event) => setSystemPrompt(event.target.value)}
              />
            </div>

            <div className="grid gap-2">
              <label className="label">Cosavu model</label>
              <div className="segmented" role="tablist" aria-label="Cosavu model">
                {tiers.map((tier) => (
                  <button
                    key={tier}
                    type="button"
                    className={tier === modelTier ? "segmented-item active" : "segmented-item"}
                    onClick={() => setModelTier(tier)}
                  >
                    {tier.replace("stan-1.5-mini-", "")}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-2">
              <label htmlFor="prompt" className="label">Prompt</label>
              <textarea
                id="prompt"
                className="textarea min-h-32"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
              />
            </div>

            <div className="flex flex-wrap gap-2">
              <button className="button primary" type="submit" disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                Run comparison
              </button>
              <button className="button secondary" type="button" onClick={() => setPrompt(defaultPrompt)}>
                Reset prompt
              </button>
            </div>
          </form>
        </section>

        {error ? <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{error}</div> : null}

        <section className="grid gap-4 xl:grid-cols-2">
          <ResultCard
            title="Direct"
            eyebrow="Cosavu direct"
            prompt={result?.prompt}
            result={result?.raw}
            tab={directTab}
            onTabChange={setDirectTab}
          />
          <ResultCard
            title="Optimized"
            eyebrow="Cosavu optimized"
            prompt={result?.optimizedPrompt}
            result={result?.optimized}
            tab={optimizedTab}
            onTabChange={setOptimizedTab}
            cosavu={result?.cosavu}
          />
        </section>

        <section className="grid gap-4 lg:grid-cols-[1fr_1fr]">
          <div className="rounded-lg border bg-card p-4">
            <div className="mb-3 flex items-center gap-2">
              <Table2 className="h-4 w-4 text-muted-foreground" />
              <p className="eyebrow mb-0">Cosavu generated parameters sent to chat</p>
            </div>
            <pre className="code-box">{JSON.stringify(result?.cosavu.params ?? {}, null, 2)}</pre>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <div className="mb-3 flex items-center gap-2">
              <Brain className="h-4 w-4 text-muted-foreground" />
              <p className="eyebrow mb-0">Cosavu notes</p>
            </div>
            <pre className="code-box">{result?.cosavu.notes ?? "No run yet."}</pre>
          </div>
        </section>

        {savings ? (
          <div className="rounded-lg border bg-card p-4 text-sm text-muted-foreground">
            Cosavu saved <strong className="text-foreground">{savings.saved}</strong> prompt tokens before the optimized chat call
            {" "}(<strong className="text-foreground">{savings.pct.toFixed(1)}%</strong> based on ContextAPI estimates).
          </div>
        ) : null}

        {costSavings ? (
          <div className="rounded-lg border bg-card p-4 text-sm text-muted-foreground">
            Direct cost <strong className="text-foreground">{formatUsd(costSavings.direct)}</strong>
            {" · "}Optimized cost <strong className="text-foreground">{formatUsd(costSavings.optimized)}</strong>
            {" · "}
            {costSavings.saved >= 0 ? "Saved" : "Extra"}{" "}
            <strong className="text-foreground">{formatUsd(Math.abs(costSavings.saved))}</strong>
            {" "}(<strong className="text-foreground">{costSavings.pct.toFixed(1)}%</strong>).
          </div>
        ) : null}
      </div>
    </main>
  );
}

function ResultCard({
  eyebrow,
  title,
  prompt,
  result,
  tab,
  onTabChange,
  cosavu,
}: {
  eyebrow: string;
  title: string;
  prompt?: string;
  result?: CompareResult["raw"];
  tab: "answer" | "thinking";
  onTabChange: (tab: "answer" | "thinking") => void;
  cosavu?: CompareResult["cosavu"];
}) {
  const content = tab === "answer" ? result?.answer : result?.thinking;

  return (
    <article className="overflow-hidden rounded-lg border bg-card">
      <div className="flex items-start justify-between gap-3 border-b bg-muted/50 px-4 py-4">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h2 className="text-xl font-semibold">{title}</h2>
        </div>
        <span className="badge">{result ? "Ready" : "Idle"}</span>
      </div>
      <div className="grid gap-3 p-4">
        <div className="prompt-box">{prompt || "Prompt appears here after a run."}</div>
        <div className="tabs">
          <button className={tab === "answer" ? "tab active" : "tab"} onClick={() => onTabChange("answer")} type="button">
            Answer
          </button>
          <button className={tab === "thinking" ? "tab active" : "tab"} onClick={() => onTabChange("thinking")} type="button">
            Thinking
          </button>
        </div>
        <div className={tab === "thinking" ? "markdown markdown-dark" : "markdown"}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || "Response appears here."}</ReactMarkdown>
        </div>
        <MetricTable result={result} cosavu={cosavu} />
      </div>
    </article>
  );
}

function MetricTable({ result, cosavu }: { result?: CompareResult["raw"]; cosavu?: CompareResult["cosavu"] }) {
  const rows = [
    cosavu ? ["Cosavu latency", `${Math.round(cosavu.latencyMs)} ms`] : null,
    cosavu ? ["Cosavu input", tokenText(cosavu.totalOriginalTokens)] : null,
    cosavu ? ["Cosavu output", tokenText(cosavu.totalOptimizedTokens)] : null,
    ["Chat input", tokenText(result?.usage.inputTokens)],
    ["Chat output (incl. reasoning)", tokenText(result?.usage.outputTokens)],
    ["Chat reasoning", tokenText(result?.usage.reasoningTokens)],
    ["Chat total", tokenText(result?.usage.totalTokens)],
    ["Chat cost", result ? formatUsd(result.costUsd) : "-"],
    ["Chat latency", result ? `${Math.round(result.latencyMs)} ms` : "-"],
    ["temperature", result?.request.temperature ?? "-"],
    ["top_p", result?.request.top_p ?? "-"],
    ["reasoning effort", result?.request.reasoning_effort ?? "-"],
  ].filter(Boolean) as [string, string | number][];

  return (
    <div className="metric-table">
      {rows.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function tokenText(value?: number | null) {
  return value === null || value === undefined ? "-" : `${value} tokens`;
}

function formatUsd(value: number) {
  if (!Number.isFinite(value) || value === 0) return "$0.000000";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  return `$${value.toFixed(4)}`;
}
