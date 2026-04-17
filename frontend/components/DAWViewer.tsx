"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { audioUrl } from "@/lib/api";
import { trackColor, fmtTime } from "@/lib/constants";
import type { ResultData, RoleData } from "@/lib/types";

interface Props {
  result: ResultData | null;
  roles: RoleData[];
}

const LANE_HEIGHT = 36;
const LABEL_WIDTH = 110;
const PX_PER_TICK = 0.15;

export default function DAWViewer({ result, roles }: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const ticksPerBeat = result?.ticks_per_beat || 480;
  const tempo = result?.tempo || 120;
  const totalMeasures = result?.total_measures || 0;
  const beatsPerMeasure = result?.time_sig?.[0] || 4;
  const ticksPerMeasure = ticksPerBeat * beatsPerMeasure;
  const totalTicks = totalMeasures * ticksPerMeasure;
  const laneWidth = totalTicks * PX_PER_TICK;

  const trackEntries = result
    ? Object.entries(result.track_notes).filter(([, notes]) => notes.length > 0)
    : [];

  const secondsToX = useCallback(
    (s: number) => {
      const ticksPerSecond = (tempo / 60) * ticksPerBeat;
      return s * ticksPerSecond * PX_PER_TICK + LABEL_WIDTH;
    },
    [tempo, ticksPerBeat]
  );

  const animate = useCallback(() => {
    if (!audioRef.current || !playheadRef.current) return;
    const t = audioRef.current.currentTime;
    setCurrentTime(t);
    playheadRef.current.style.left = `${secondsToX(t)}px`;
    playheadRef.current.style.display = "block";
    if (!audioRef.current.paused) {
      animRef.current = requestAnimationFrame(animate);
    }
  }, [secondsToX]);

  const play = () => {
    if (!audioRef.current) return;
    if (audioRef.current.paused) {
      audioRef.current.play();
      setPlaying(true);
      animRef.current = requestAnimationFrame(animate);
    } else {
      audioRef.current.pause();
      setPlaying(false);
      cancelAnimationFrame(animRef.current);
    }
  };

  const stop = () => {
    if (!audioRef.current) return;
    audioRef.current.pause();
    audioRef.current.currentTime = 0;
    setPlaying(false);
    setCurrentTime(0);
    cancelAnimationFrame(animRef.current);
    if (playheadRef.current) {
      playheadRef.current.style.left = `${LABEL_WIDTH}px`;
    }
  };

  useEffect(() => {
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  if (!result || trackEntries.length === 0) return null;

  return (
    <div className="card">
      {/* Transport */}
      <div className="flex items-center gap-2 mb-3 text-sm">
        <button onClick={play} className="btn-sm">{playing ? "\u23F8" : "\u25B6"}</button>
        <button onClick={stop} className="btn-sm">{"\u25A0"}</button>
        <span className="text-dim font-mono text-xs">
          {fmtTime(currentTime)} / {fmtTime(duration)}
        </span>
        <div className="flex-1" />
        <span className="text-dim text-[10px]">
          {trackEntries.length} tracks &middot; {result.total_notes} notes
        </span>
      </div>

      {/* Audio */}
      {result.audio_url && (
        <audio
          ref={audioRef}
          src={audioUrl(result.audio_url)}
          onLoadedMetadata={() => setDuration(audioRef.current?.duration || 0)}
          onEnded={stop}
        />
      )}

      {/* Viewport */}
      <div
        ref={viewportRef}
        className="relative overflow-x-auto overflow-y-hidden border border-border rounded-md bg-surface2"
        style={{ minHeight: Math.max(120, trackEntries.length * LANE_HEIGHT + 24) }}
      >
        {/* Ruler */}
        <div className="sticky top-0 z-10 h-6 bg-surface2 border-b border-border flex text-[9px] text-dim select-none">
          {Array.from({ length: totalMeasures }, (_, i) => (
            <div
              key={i}
              className="absolute h-6 flex items-end px-1 pb-0.5 border-l border-border"
              style={{ left: LABEL_WIDTH + i * ticksPerMeasure * PX_PER_TICK }}
            >
              {i + 1}
            </div>
          ))}
        </div>

        {/* Lanes */}
        <div className="relative">
          {trackEntries.map(([name, notes]) => {
            const color = trackColor(name, roles);
            const allPitches = notes.map((n) => n.pitch);
            const minP = Math.min(...allPitches);
            const maxP = Math.max(...allPitches);
            const range = Math.max(1, maxP - minP);

            return (
              <div key={name} className="flex border-b border-border relative" style={{ height: LANE_HEIGHT }}>
                {/* Label */}
                <div className="sticky left-0 z-20 bg-surface border-r border-border flex items-center px-2 text-[10px] font-semibold gap-1.5"
                     style={{ width: LABEL_WIDTH, minWidth: LABEL_WIDTH }}>
                  <span className="w-2 h-2 rounded-sm flex-shrink-0" style={{ background: color }} />
                  <span className="truncate">{name}</span>
                </div>

                {/* Notes */}
                <div className="relative flex-1" style={{ width: laneWidth }}>
                  {notes.map((n, i) => {
                    const x = n.start * PX_PER_TICK;
                    const w = Math.max(2, (n.end - n.start) * PX_PER_TICK);
                    const yPct = 1 - (n.pitch - minP) / range;
                    const top = 4 + yPct * (LANE_HEIGHT - 10);
                    return (
                      <div
                        key={i}
                        className="absolute rounded-sm"
                        style={{
                          left: x,
                          top,
                          width: w,
                          height: 3,
                          background: color,
                          opacity: 0.5 + (n.velocity / 127) * 0.5,
                        }}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Playhead */}
        <div
          ref={playheadRef}
          className="absolute top-0 bottom-0 w-0.5 bg-danger z-30 pointer-events-none hidden"
          style={{ left: LABEL_WIDTH, boxShadow: "0 0 6px rgba(220,38,38,0.5)" }}
        />
      </div>
    </div>
  );
}
