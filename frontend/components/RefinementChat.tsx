"use client";

import { useState } from "react";
import { refine } from "@/lib/api";
import type { RefineResponse, ResultData } from "@/lib/types";

interface Props {
  hasResult: boolean;
  onRefineResult: (res: RefineResponse) => void;
}

export default function RefinementChat({ hasResult, onRefineResult }: Props) {
  const [feedback, setFeedback] = useState("");
  const [history, setHistory] = useState<{ user: string; response: string }[]>([]);
  const [loading, setLoading] = useState(false);

  if (!hasResult) return null;

  const send = async () => {
    if (!feedback.trim() || loading) return;
    const msg = feedback;
    setFeedback("");
    setLoading(true);

    try {
      const res = await refine(msg, 0, 0);
      setHistory((h) => [...h, { user: msg, response: `Applied: ${JSON.stringify(res.feedback_applied)}` }]);
      onRefineResult(res);
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : "Refinement failed";
      setHistory((h) => [...h, { user: msg, response: errMsg }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2 className="card-title">Refine</h2>
      {history.length > 0 && (
        <div className="flex flex-col gap-2 mb-3 max-h-40 overflow-y-auto text-xs">
          {history.map((h, i) => (
            <div key={i}>
              <div className="text-accent font-semibold">{h.user}</div>
              <div className="text-dim">{h.response}</div>
            </div>
          ))}
        </div>
      )}
      <div className="flex gap-2">
        <input
          className="input-field flex-1"
          placeholder="e.g. remove brass, make quieter, more strings..."
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          disabled={loading}
        />
        <button className="btn-primary px-4" onClick={send} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
