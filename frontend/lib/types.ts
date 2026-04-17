export interface StyleOption {
  name: string;
  description: string;
}

export interface EnsembleOption {
  name: string;
  instruments: string[];
}

export interface OptionsResponse {
  styles: StyleOption[];
  ensembles: EnsembleOption[];
}

export interface NoteData {
  pitch: number;
  start: number;
  end: number;
  velocity: number;
}

export interface ChordData {
  root: string;
  quality: string;
  measure: number;
  label: string;
}

export interface AnalysisData {
  key: string;
  tempo: number;
  time_sig: string;
  total_measures: number;
  ticks_per_beat: number;
  melody_notes: NoteData[];
  bass_notes: NoteData[];
  inner_notes: NoteData[];
  chords: ChordData[];
  phrase_boundaries: number[];
  note_counts: {
    melody: number;
    bass: number;
    inner: number;
    total: number;
  };
}

export interface RoleData {
  instrument: string;
  role: string;
  family: string;
  velocity_scale: number;
  doubles: string | null;
}

export interface PlanData {
  style: string;
  ensemble: string;
  roles: RoleData[];
  kb_advice?: string[];
}

export interface TrackData {
  name: string;
  notes: number;
  channel: number;
}

export interface TrackNotes {
  [trackName: string]: NoteData[];
}

export interface ResultData {
  total_notes: number;
  num_tracks: number;
  tracks: TrackData[];
  track_notes: TrackNotes;
  download_url: string;
  audio_url: string | null;
  tempo: number;
  ticks_per_beat: number;
  time_sig: number[];
  total_measures: number;
}

export interface OrchestrateResponse {
  analysis: AnalysisData;
  plan: PlanData;
  result: ResultData;
}

export interface ReharmonizeResponse {
  analysis: AnalysisData;
  result: {
    total_notes: number;
    num_tracks: number;
    tracks: TrackData[];
    chords: ChordData[];
    download_url: string;
  };
}

export interface RefineResponse {
  feedback_applied: Record<string, unknown>;
  measures: { start: number; end: number };
  plan: PlanData;
  result: ResultData;
}

export interface ChatResponse {
  answer: string;
  source: "llm+rag" | "rag" | "fallback";
}

export interface HealthResponse {
  status: string;
  python: string;
  torch: string;
  device: string;
  audio_renderer: boolean;
  output_dir: string;
  generated_files: number;
}

export interface LogEntry {
  ts: string;
  level: string;
  logger: string;
  message: string;
  exc: string | null;
}

export interface LogsResponse {
  logs: LogEntry[];
  total: number;
}

export interface SoundFont {
  name: string;
  size_mb: number;
  active?: boolean;
}
