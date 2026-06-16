export type ChatResult = {
  answer: string;
  thinking: string;
  request: {
    temperature: number;
    top_p: number;
    reasoning_format: string | null;
    reasoning_effort: string | null;
    max_completion_tokens: number | null;
  };
  usage: {
    inputTokens: number | null;
    outputTokens: number | null;
    totalTokens: number | null;
  };
  latencyMs: number;
};

export type CompareResult = {
  prompt: string;
  optimizedPrompt: string;
  cosavu: {
    modelTier: string;
    notes: string;
    params: {
      generated: Record<string, unknown>;
      model: Record<string, unknown>;
    };
    totalOriginalTokens: number;
    totalOptimizedTokens: number;
    latencyMs: number;
  };
  raw: ChatResult;
  optimized: ChatResult;
};
