import type {
  OptionsResponse,
  OrchestrateResponse,
  ReharmonizeResponse,
  RefineResponse,
  ChatResponse,
  HealthResponse,
  LogsResponse,
  SoundFont,
} from "./types";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `API error ${res.status}`);
  }
  return res.json();
}

export function getOptions() {
  return fetchJSON<OptionsResponse>("/api/options");
}

export function getHealth() {
  return fetchJSON<HealthResponse>("/api/health");
}

export function getLogs(level = "INFO", limit = 100) {
  return fetchJSON<LogsResponse>(`/api/logs?level=${level}&limit=${limit}`);
}

export function orchestrate(file: File, style: string, ensemble: string, useLLM: boolean) {
  const form = new FormData();
  form.append("file", file);
  form.append("style", style);
  form.append("ensemble", ensemble);
  form.append("use_llm", String(useLLM));
  return fetchJSON<OrchestrateResponse>("/api/orchestrate", { method: "POST", body: form });
}

export function reharmonize(file: File, style: string) {
  const form = new FormData();
  form.append("file", file);
  form.append("style", style);
  return fetchJSON<ReharmonizeResponse>("/api/reharmonize", { method: "POST", body: form });
}

export function refine(feedback: string, measureStart: number, measureEnd: number) {
  const form = new FormData();
  form.append("feedback", feedback);
  form.append("measure_start", String(measureStart));
  form.append("measure_end", String(measureEnd));
  return fetchJSON<RefineResponse>("/api/refine", { method: "POST", body: form });
}

export function chat(question: string) {
  const form = new FormData();
  form.append("question", question);
  return fetchJSON<ChatResponse>("/api/chat", { method: "POST", body: form });
}

export function uploadSoundFont(file: File) {
  const form = new FormData();
  form.append("file", file);
  return fetchJSON<SoundFont>("/api/soundfont", { method: "POST", body: form });
}

export function listSoundFonts() {
  return fetchJSON<{ soundfonts: SoundFont[] }>("/api/soundfonts");
}

export function selectSoundFont(name: string) {
  const form = new FormData();
  form.append("name", name);
  return fetchJSON<{ active: string }>("/api/soundfont/select", { method: "POST", body: form });
}

export function downloadUrl(path: string) {
  return `${API}${path}`;
}

export function audioUrl(path: string) {
  return `${API}${path}`;
}
