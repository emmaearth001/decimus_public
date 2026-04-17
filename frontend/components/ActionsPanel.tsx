"use client";

import { downloadUrl } from "@/lib/api";

interface Props {
  hasFile: boolean;
  loading: boolean;
  resultDownloadUrl: string | null;
  onOrchestrate: () => void;
  onReharmonize: () => void;
}

export default function ActionsPanel({ hasFile, loading, resultDownloadUrl, onOrchestrate, onReharmonize }: Props) {
  return (
    <div className="card">
      <h2 className="card-title">Actions</h2>
      <button
        className="btn-primary w-full"
        disabled={!hasFile || loading}
        onClick={onOrchestrate}
      >
        {loading ? "Processing..." : "Orchestrate"}
      </button>
      <button
        className="btn-secondary w-full mt-2"
        disabled={!hasFile || loading}
        onClick={onReharmonize}
      >
        Reharmonize (Strings)
      </button>
      {resultDownloadUrl && (
        <a
          href={downloadUrl(resultDownloadUrl)}
          className="btn-download w-full mt-3 block text-center"
          download
        >
          Download MIDI
        </a>
      )}
    </div>
  );
}
