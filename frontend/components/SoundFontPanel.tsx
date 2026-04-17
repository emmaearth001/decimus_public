"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { uploadSoundFont, listSoundFonts, selectSoundFont } from "@/lib/api";
import type { SoundFont } from "@/lib/types";

export default function SoundFontPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fonts, setFonts] = useState<SoundFont[]>([]);
  const [uploading, setUploading] = useState(false);

  const loadFonts = useCallback(async () => {
    try {
      const data = await listSoundFonts();
      setFonts(data.soundfonts);
    } catch {}
  }, []);

  useEffect(() => { loadFonts(); }, [loadFonts]);

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".sf2")) return;
    setUploading(true);
    try {
      await uploadSoundFont(file);
      await loadFonts();
    } catch {}
    setUploading(false);
  };

  return (
    <div className="card">
      <h2 className="card-title">Sound Library</h2>
      <div className="text-[11px] text-dim mb-2">Upload a .sf2 SoundFont for better instrument sounds.</div>
      <div
        className="border-2 border-dashed border-border rounded-lg p-4 text-center cursor-pointer hover:border-accent transition-all text-xs text-dim"
        onClick={() => inputRef.current?.click()}
      >
        {uploading ? "Uploading..." : "Drop .sf2 file or click"}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".sf2"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f); }}
      />
      {fonts.length > 0 && (
        <div className="mt-2 flex flex-col gap-1 text-[11px]">
          {fonts.map((f) => (
            <button
              key={f.name}
              onClick={() => selectSoundFont(f.name).then(loadFonts)}
              className={`text-left px-2 py-1 rounded ${f.active ? "bg-accent/10 text-accent" : "text-dim hover:text-text"}`}
            >
              {f.name} ({f.size_mb} MB) {f.active && "  \u2713"}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
