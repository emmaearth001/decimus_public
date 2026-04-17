"use client";

import { useState, useCallback } from "react";
import Header from "@/components/Header";
import DropZone from "@/components/DropZone";
import SettingsPanel from "@/components/SettingsPanel";
import ActionsPanel from "@/components/ActionsPanel";
import StatusBar from "@/components/StatusBar";
import DAWViewer from "@/components/DAWViewer";
import RefinementChat from "@/components/RefinementChat";
import SoundFontPanel from "@/components/SoundFontPanel";
import LogPanel from "@/components/LogPanel";
import { orchestrate, reharmonize } from "@/lib/api";
import type { ResultData, RoleData, RefineResponse } from "@/lib/types";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [style, setStyle] = useState("tchaikovsky");
  const [ensemble, setEnsemble] = useState("full");
  const [useLLM, setUseLLM] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ message: "", type: "info" as "info" | "error" | "success" });
  const [result, setResult] = useState<ResultData | null>(null);
  const [roles, setRoles] = useState<RoleData[]>([]);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

  const handleOrchestrate = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setStatus({ message: "Orchestrating...", type: "info" });
    try {
      const res = await orchestrate(file, style, ensemble, useLLM);
      setResult(res.result);
      setRoles(res.plan.roles);
      setDownloadUrl(res.result.download_url);
      setStatus({
        message: `Done: ${res.result.total_notes} notes, ${res.result.num_tracks} tracks`,
        type: "success",
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Orchestration failed";
      setStatus({ message: msg, type: "error" });
    } finally {
      setLoading(false);
    }
  }, [file, style, ensemble, useLLM]);

  const handleReharmonize = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setStatus({ message: "Reharmonizing...", type: "info" });
    try {
      const res = await reharmonize(file, style);
      setDownloadUrl(res.result.download_url);
      setStatus({
        message: `Reharmonized: ${res.result.total_notes} notes, ${res.result.num_tracks} tracks`,
        type: "success",
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Reharmonization failed";
      setStatus({ message: msg, type: "error" });
    } finally {
      setLoading(false);
    }
  }, [file, style]);

  const handleRefine = useCallback((res: RefineResponse) => {
    setResult(res.result);
    setRoles(res.plan.roles);
    setDownloadUrl(res.result.download_url);
    setStatus({
      message: `Refined: ${res.result.total_notes} notes, ${res.result.num_tracks} tracks`,
      type: "success",
    });
  }, []);

  return (
    <>
      <Header />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 grid grid-cols-1 md:grid-cols-[280px_1fr] lg:grid-cols-[300px_1fr] gap-6 pb-16">
        {/* Left: Controls */}
        <div className="flex flex-col gap-4">
          <DropZone onFile={setFile} />
          <SettingsPanel
            style={style}
            ensemble={ensemble}
            useLLM={useLLM}
            onStyleChange={setStyle}
            onEnsembleChange={setEnsemble}
            onLLMChange={setUseLLM}
          />
          <ActionsPanel
            hasFile={!!file}
            loading={loading}
            resultDownloadUrl={downloadUrl}
            onOrchestrate={handleOrchestrate}
            onReharmonize={handleReharmonize}
          />
          <SoundFontPanel />
          <StatusBar message={status.message} type={status.type} />
        </div>

        {/* Center: Results */}
        <div className="flex flex-col gap-6">
          <DAWViewer result={result} roles={roles} />
          <RefinementChat hasResult={!!result} onRefineResult={handleRefine} />
        </div>

      </div>

      <LogPanel />
    </>
  );
}
