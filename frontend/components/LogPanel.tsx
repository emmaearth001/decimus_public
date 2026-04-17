"use client";

import { useCallback, useEffect, useState } from "react";
import { getLogs, getHealth } from "@/lib/api";
import type { LogEntry } from "@/lib/types";

export default function LogPanel() {
  const [expanded, setExpanded] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [level, setLevel] = useState("INFO");
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [healthInfo, setHealthInfo] = useState("");

  const fetchLogs = useCallback(async () => {
    try {
      const data = await getLogs(level, 100);
      setLogs(data.logs);
    } catch {}
  }, [level]);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const h = await getHealth();
        setHealthy(true);
        setHealthInfo(`Python ${h.python} | ${h.device} | ${h.generated_files} files`);
      } catch {
        setHealthy(false);
        setHealthInfo("Server unreachable");
      }
    };
    checkHealth();
    const hInterval = setInterval(checkHealth, 10000);
    return () => clearInterval(hInterval);
  }, []);

  useEffect(() => {
    if (!expanded) return;
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, [expanded, fetchLogs]);

  return (
    <div className={`fixed bottom-0 left-0 right-0 z-50 bg-surface border-t border-border transition-all ${expanded ? "h-64" : "h-8"}`}>
      <div
        className="h-8 px-4 flex items-center gap-3 cursor-pointer text-xs select-none"
        onClick={() => setExpanded(!expanded)}
      >
        <span>{expanded ? "\u25BC" : "\u25B2"}</span>
        <span>Logs</span>
        <span className="bg-surface2 text-dim text-[10px] px-1.5 rounded">{logs.length}</span>
        <div
          className={`w-2 h-2 rounded-full ${healthy === true ? "bg-success" : healthy === false ? "bg-danger" : "bg-dim"}`}
          title={healthInfo}
        />
        <div className="flex-1" />
        {expanded && (
          <>
            <select
              className="bg-surface2 text-dim text-[10px] border border-border rounded px-1 py-0.5"
              value={level}
              onChange={(e) => setLevel(e.target.value)}
              onClick={(e) => e.stopPropagation()}
            >
              <option value="DEBUG">ALL</option>
              <option value="INFO">INFO+</option>
              <option value="WARNING">WARN+</option>
              <option value="ERROR">ERROR</option>
            </select>
            <button
              className="text-[10px] text-dim hover:text-text"
              onClick={(e) => { e.stopPropagation(); setLogs([]); }}
            >
              Clear
            </button>
          </>
        )}
      </div>
      {expanded && (
        <div className="overflow-y-auto h-[calc(100%-32px)] px-4 pb-2 font-mono text-[11px] leading-5">
          {logs.map((log, i) => (
            <div key={i} className={`${log.level === "ERROR" ? "text-danger" : log.level === "WARNING" ? "text-yellow-600" : "text-dim"}`}>
              <span className="text-dim/60">{log.ts}</span> <span className="font-semibold">{log.level.padEnd(7)}</span> {log.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
