"use client";

import { useEffect, useState } from "react";
import { getOptions } from "@/lib/api";
import type { StyleOption, EnsembleOption } from "@/lib/types";

interface Props {
  style: string;
  ensemble: string;
  useLLM: boolean;
  onStyleChange: (v: string) => void;
  onEnsembleChange: (v: string) => void;
  onLLMChange: (v: boolean) => void;
}

export default function SettingsPanel({ style, ensemble, useLLM, onStyleChange, onEnsembleChange, onLLMChange }: Props) {
  const [styles, setStyles] = useState<StyleOption[]>([]);
  const [ensembles, setEnsembles] = useState<EnsembleOption[]>([]);

  useEffect(() => {
    getOptions().then((data) => {
      setStyles(data.styles);
      setEnsembles(data.ensembles);
    }).catch(() => {});
  }, []);

  return (
    <>
      <div className="card">
        <h2 className="card-title">Settings</h2>
        <div className="flex flex-col gap-2">
          <label className="label-sm">Composer</label>
          <select className="select-field" value={style} onChange={(e) => onStyleChange(e.target.value)}>
            {styles.map((s) => (
              <option key={s.name} value={s.name}>{s.name}</option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-2 mt-3">
          <label className="label-sm">Ensemble</label>
          <select className="select-field" value={ensemble} onChange={(e) => onEnsembleChange(e.target.value)}>
            {ensembles.map((e) => (
              <option key={e.name} value={e.name}>{e.name}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="card">
        <label className="flex items-center gap-2 cursor-pointer text-sm">
          <input
            type="checkbox"
            checked={useLLM}
            onChange={(e) => onLLMChange(e.target.checked)}
            className="accent-accent"
          />
          <span>AI-Powered (Decimus LLM)</span>
        </label>
        <div className="text-[10px] text-dim mt-1">
          Uses fine-tuned LLM for smarter instrument choices. First call may take ~60s (cold start).
        </div>
      </div>
    </>
  );
}
