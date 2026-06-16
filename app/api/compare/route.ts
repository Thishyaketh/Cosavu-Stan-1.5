import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const ROOT_DIR = path.resolve(process.cwd(), "..");
const CHAT_MODEL = process.env.CHAT_MODEL || process.env.GROQ_MODEL || "openai/gpt-oss-120b";
const COSAVU_URL = (process.env.CONTEXTAPI_URL || "https://api.cosavu.com").replace(/\/$/, "");
const COSAVU_ORIGIN = process.env.CONTEXTAPI_ORIGIN || "https://cosavu.com";

type ChatPayload = {
  model: string;
  messages: { role: "system" | "user"; content: string }[];
  temperature: number;
  top_p: number;
  reasoning_format?: string;
  reasoning_effort?: string;
  max_completion_tokens?: number;
};

await loadRootEnv();

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const prompt = String(body.prompt || "").trim();
    const systemPrompt = String(body.systemPrompt || "Answer clearly and directly.").trim();
    const modelTier = String(body.modelTier || "stan-1.5-mini-predictive");

    if (!prompt) {
      return NextResponse.json({ error: "Prompt is required." }, { status: 400 });
    }
    if (!process.env.GROQ_API_KEY) {
      return NextResponse.json({ error: "Model API key is missing in .env." }, { status: 500 });
    }

    const cosavu = await optimizeWithCosavu(prompt, modelTier);
    const optimizedPrompt = reassemble(cosavu);
    const cosavuParams = parseCosavuParams(cosavu.optimization_notes);

    const directPayload = buildChatPayload({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ],
      params: { temperature: 0.7, top_p: 0.9, reasoning_effort: "high" },
    });

    const optimizedPayload = buildChatPayload({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: optimizedPrompt },
      ],
      params: { ...cosavuParams.model, reasoning_effort: "high" },
    });

    const [direct, optimized] = await Promise.all([
      callChat(directPayload),
      callChat(optimizedPayload),
    ]);

    return NextResponse.json({
      prompt,
      optimizedPrompt,
      cosavu: {
        modelTier,
        notes: sanitizeNotes(cosavu.optimization_notes),
        params: cosavuParams,
        totalOriginalTokens: cosavu.total_original_tokens,
        totalOptimizedTokens: cosavu.total_optimized_tokens,
        latencyMs: cosavu.latency_ms,
      },
      raw: formatChatResult(direct, directPayload),
      optimized: formatChatResult(optimized, optimizedPayload),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Comparison failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

function buildChatPayload({
  messages,
  params,
}: {
  messages: ChatPayload["messages"];
  params: Record<string, unknown>;
}): ChatPayload {
  const payload: ChatPayload = {
    model: CHAT_MODEL,
    messages,
    temperature: numberOr(params.temperature, 0.7),
    top_p: numberOr(params.top_p, 0.9),
    reasoning_format: "parsed",
  };
  if (typeof params.reasoning_effort === "string") {
    payload.reasoning_effort = params.reasoning_effort;
  }
  if (Number.isFinite(params.max_completion_tokens)) {
    payload.max_completion_tokens = Math.round(Number(params.max_completion_tokens));
  }
  return payload;
}

async function optimizeWithCosavu(prompt: string, modelTier: string) {
  const headers: Record<string, string> = {
    Accept: "application/json",
    "Content-Type": "application/json",
    Origin: COSAVU_ORIGIN,
    "User-Agent": "Mozilla/5.0 Cosavu-Shadcn-Demo/1.0",
  };
  if (process.env.COSAVU_API_KEY) {
    headers["X-API-Token"] = process.env.COSAVU_API_KEY;
  }

  const response = await fetch(`${COSAVU_URL}/optimize`, {
    method: "POST",
    headers,
    body: JSON.stringify({ prompt, model_tier: modelTier }),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Cosavu ${response.status}: ${text}`);
  }
  return JSON.parse(text);
}

async function callChat(payload: ChatPayload) {
  const started = performance.now();
  let sentPayload = payload;
  let { response, text } = await postChat(sentPayload);

  if (!response.ok && shouldRetryWithoutReasoning(text)) {
    sentPayload = { ...payload };
    delete sentPayload.reasoning_format;
    delete sentPayload.reasoning_effort;
    ({ response, text } = await postChat(sentPayload));
  }

  const latencyMs = performance.now() - started;
  if (!response.ok) {
    throw new Error(`Chat model ${response.status}: ${text}`);
  }
  return { ...JSON.parse(text), _latency_ms: latencyMs, _request_payload: sentPayload };
}

async function postChat(payload: ChatPayload) {
  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return { response, text: await response.text() };
}

function formatChatResult(json: any, requestPayload: ChatPayload) {
  const usage = json.usage || json.x_groq?.usage || {};
  const actualPayload = json._request_payload || requestPayload;
  const reasoningTokens =
    usage.completion_tokens_details?.reasoning_tokens ??
    usage.reasoning_tokens ??
    0;
  const baseCompletion = usage.completion_tokens ?? usage.output_tokens ?? null;
  const baseInput = usage.prompt_tokens ?? usage.input_tokens ?? null;
  const baseTotal = usage.total_tokens ?? null;
  const outputTokens = baseCompletion === null ? null : baseCompletion + reasoningTokens;
  const totalTokens =
    baseTotal === null ? null : baseTotal + reasoningTokens;
  return {
    answer: json.choices?.[0]?.message?.content || "",
    thinking: extractThinking(json),
    request: {
      temperature: actualPayload.temperature,
      top_p: actualPayload.top_p,
      reasoning_format: actualPayload.reasoning_format || null,
      reasoning_effort: actualPayload.reasoning_effort || null,
      max_completion_tokens: actualPayload.max_completion_tokens || null,
    },
    usage: {
      inputTokens: baseInput,
      outputTokens,
      totalTokens,
      reasoningTokens,
    },
    latencyMs: json._latency_ms,
  };
}

function extractThinking(json: any) {
  const message = json.choices?.[0]?.message || {};
  const candidates = [
    message.reasoning,
    message.reasoning_content,
    message.thinking,
    message.analysis,
    json.x_groq?.reasoning,
    json.reasoning,
  ];
  for (const candidate of candidates) {
    if (!candidate) continue;
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
    if (typeof candidate === "object") return JSON.stringify(candidate, null, 2);
  }
  return "No separate thinking payload was returned for this response.";
}

function parseCosavuParams(notes: string) {
  const clean = sanitizeNotes(notes);
  const temperature = matchNumber(clean, /\btemp=([0-9.]+)/);
  const topP = matchNumber(clean, /\btop_p=([0-9.]+)/);
  const maxNewTokens = matchNumber(clean, /\bmax_new_tokens=([0-9]+)/);
  const mode = matchText(clean, /\bmode=([A-Z_]+)/);
  const reasoningEffort = toReasoningEffort(mode);

  const model: Record<string, unknown> = {};
  if (Number.isFinite(temperature)) model.temperature = temperature;
  if (Number.isFinite(topP)) model.top_p = topP;
  if (Number.isFinite(maxNewTokens)) model.max_completion_tokens = Math.round(Number(maxNewTokens));
  if (reasoningEffort) model.reasoning_effort = reasoningEffort;

  return {
    generated: {
      temperature,
      top_p: topP,
      max_new_tokens: maxNewTokens,
      reasoning_mode: mode,
      reasoning_effort: reasoningEffort,
    },
    model,
  };
}

function toReasoningEffort(mode: string | null) {
  if (!mode) return null;
  if (/DEEP|PREDICT|HIGH/.test(mode)) return "high";
  if (/FAST|LIGHT|LOW/.test(mode)) return "low";
  return "medium";
}

function sanitizeNotes(notes: string) {
  return String(notes || "-")
    .replace(/\s*rewriter=[^|]+(?=\s*\|)/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function reassemble(ir: any) {
  return (ir.blocks || []).map((block: any) => block.content || "").join("\n\n").trim();
}

function numberOr(value: unknown, fallback: number) {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function matchNumber(text: string, pattern: RegExp) {
  const m = text.match(pattern);
  return m ? Number(m[1]) : null;
}

function matchText(text: string, pattern: RegExp) {
  const m = text.match(pattern);
  return m ? m[1] : null;
}

function shouldRetryWithoutReasoning(text: string) {
  return /reasoning_(format|effort)|unsupported|extra fields|unrecognized/i.test(text || "");
}

async function loadRootEnv() {
  for (const filePath of [path.join(ROOT_DIR, ".env"), path.join(process.cwd(), ".env.local")]) {
    try {
      const env = await readFile(filePath, "utf8");
      for (const line of env.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const match = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
        if (!match) continue;
        const [, key, rawValue] = match;
        if (process.env[key]) continue;
        process.env[key] = rawValue.trim().replace(/^["']|["']$/g, "");
      }
    } catch {
      // Optional env file.
    }
  }
}
