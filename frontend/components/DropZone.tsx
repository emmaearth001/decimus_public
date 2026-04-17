"use client";

import { useCallback, useRef, useState } from "react";
import { validateMidi } from "@/lib/constants";

interface Props {
  onFile: (file: File) => void;
}

export default function DropZone({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragover, setDragover] = useState(false);
  const [filename, setFilename] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      const bytes = new Uint8Array(await file.arrayBuffer());
      const err = validateMidi(bytes, file.name);
      if (err) {
        setError(err);
        return;
      }
      setFilename(file.name);
      onFile(file);
    },
    [onFile]
  );

  return (
    <div className="card">
      <h2 className="card-title">Input</h2>
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
          dragover ? "border-accent bg-accent/10" : "border-border"
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
        onDragLeave={() => setDragover(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragover(false);
          const f = e.dataTransfer.files[0];
          if (f) handleFile(f);
        }}
      >
        <div className="text-3xl mb-2">&#9835;</div>
        <div className="text-sm text-dim">Drop a MIDI file or click to browse</div>
        {filename && <div className="text-sm text-accent mt-2 break-all">{filename}</div>}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".mid,.midi"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      {error && <div className="text-xs text-danger mt-2">{error}</div>}
      <div className="text-[10px] text-dim mt-2">
        Accepts .mid/.midi files up to 5 MB. Piano tracks work best.
      </div>
    </div>
  );
}
