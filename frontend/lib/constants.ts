export const ROLE_COLORS: Record<string, string> = {
  melody: "#d4365a",
  doubling: "#d4365a",
  bass: "#2a7abf",
  countermelody: "#7c3aed",
  harmony: "#2aaa70",
  timpani: "#c49a1a",
  percussion: "#d97706",
};

export function trackColor(trackName: string, roles?: { instrument: string; role: string }[]): string {
  if (!roles) return "#2aaa70";
  const role = roles.find((r) => r.instrument === trackName);
  return role ? ROLE_COLORS[role.role] || "#2aaa70" : "#2aaa70";
}

export const MIDI_HEADER = [0x4d, 0x54, 0x68, 0x64]; // "MThd"

export function validateMidi(bytes: Uint8Array, filename: string): string | null {
  if (bytes.length === 0) return "Empty file. Please upload a MIDI file (.mid).";
  if (bytes.length > 5 * 1024 * 1024) return "File too large. Maximum is 5 MB.";
  if (bytes.length < 14) return "File too small to be a MIDI file.";
  for (let i = 0; i < 4; i++) {
    if (bytes[i] !== MIDI_HEADER[i]) {
      const ext = filename.split(".").pop()?.toLowerCase();
      if (["mp3", "wav", "ogg", "flac"].includes(ext || ""))
        return `'${filename}' is an audio file, not MIDI. Convert to .mid first.`;
      return `'${filename}' is not a valid MIDI file.`;
    }
  }
  return null;
}

export function fmtTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
