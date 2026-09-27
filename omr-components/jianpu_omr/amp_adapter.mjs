// Adapter from jpeditor's parsed ScoreDoc to AutoMusic Player's rich source JSON.
// This file contains no network access and is intentionally kept separate from
// the upstream jpeditor files so the latter can be replaced by a pinned build.

const SIMPLE_DIVISIONS = 48;
const STEP_TO_PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const ACCIDENTAL_OFFSET = {
  sharp: 1,
  flat: -1,
  natural: 0,
  "double-sharp": 2,
  "double-flat": -2,
};

function midiFromPitch(pitch) {
  if (!pitch || typeof pitch !== "object") return null;
  const pc = STEP_TO_PC[String(pitch.step || "").toUpperCase()];
  const octave = Number(pitch.octave);
  const alter = Number(pitch.alter || 0);
  if (pc === undefined || !Number.isFinite(octave) || !Number.isFinite(alter)) {
    return null;
  }
  const midi = 12 * (octave + 1) + pc + alter;
  return Number.isInteger(midi) && midi >= 0 && midi <= 127 ? midi : null;
}

function degreeMidi(degree, fifths, state) {
  if (!degree || !Number.isInteger(Number(degree.number))) return null;
  const number = Number(degree.number);
  if (number <= 0) return null;
  const octaveShift = Number(degree.octaveShift || 0);
  if (!Number.isInteger(octaveShift)) return null;
  const tonic = ((4 * fifths + 28) % 7 + 7) % 7;
  const degreeIndex = Math.max(1, Math.min(7, number)) - 1;
  const stepIndex = (tonic + degreeIndex) % 7;
  const wrap = Math.floor((tonic + degreeIndex) / 7);
  const steps = ["C", "D", "E", "F", "G", "A", "B"];
  const baseAlter = fifths > 0
    ? [3, 0, 4, 1, 5, 2, 6].slice(0, fifths).includes(stepIndex) ? 1 : 0
    : fifths < 0
      ? [6, 2, 5, 1, 4, 0, 3].slice(0, -fifths).includes(stepIndex) ? -1 : 0
      : 0;
  const tonicBOffset = tonic === 6 ? 1 : 0;
  const octave = 4 + octaveShift + wrap - tonicBOffset;
  const accidental = degree.accidental;
  if (accidental !== undefined && Object.prototype.hasOwnProperty.call(ACCIDENTAL_OFFSET, accidental)) {
    state.set(number, ACCIDENTAL_OFFSET[accidental]);
    if (accidental === "natural") state.set(number, 0);
  }
  const alter = baseAlter + (state.get(number) || 0);
  const midi = 12 * (octave + 1) + STEP_TO_PC[steps[stepIndex]] + alter;
  return Number.isInteger(midi) && midi >= 0 && midi <= 127 ? midi : null;
}

function elementDurationBeats(element) {
  const divisions = Number(element?.duration?.divisions);
  if (Number.isFinite(divisions) && divisions > 0) return divisions / SIMPLE_DIVISIONS;
  const beats = Number(element?.beats);
  if (Number.isFinite(beats) && beats > 0) return beats;
  return 1;
}

function isRest(element, note) {
  return element?.rest !== undefined || Number(note?.degree?.number) === 0;
}

/** Convert one parsed jpeditor ScoreDoc to the project's source JSON. */
export function scoreDocToRich(doc, options = {}) {
  const warnings = Array.isArray(options.warnings) ? [...options.warnings] : [];
  const songs = Array.isArray(doc?.songs) ? doc.songs : [];
  if (!songs.length) throw new Error("jpeditor 没有输出可解析的乐谱");
  if (songs.length > 1) warnings.push(`识别结果包含 ${songs.length} 首乐谱，当前只导入第一首`);
  const song = songs[0];
  const notes = [];
  const tracks = {};
  let nextTrack = 0;
  let durationBeats = 0;
  const parts = Array.isArray(song.parts) ? song.parts : [];
  const fifths = Number(song.key?.fifths || 0);

  for (const part of parts) {
    const partId = String(part.id || `P${nextTrack + 1}`);
    let partBeat = 0;
    const trackByVoice = new Map();
    for (const measure of Array.isArray(part.measures) ? part.measures : []) {
      const cursors = new Map();
      const accidentalStates = new Map();
      const elements = Array.isArray(measure.elements) ? measure.elements : [];
      for (const element of elements) {
        const voice = String(element.voice ?? 1);
        const staff = String(element.staff ?? 1);
        const trackKey = `${partId}:voice${voice}:staff${staff}`;
        if (!trackByVoice.has(trackKey)) {
          trackByVoice.set(trackKey, nextTrack);
          tracks[String(nextTrack)] = trackKey;
          nextTrack += 1;
        }
        const track = trackByVoice.get(trackKey);
        if (!cursors.has(voice)) cursors.set(voice, 0);
        if (!accidentalStates.has(voice)) accidentalStates.set(voice, new Map());
        const startBeat = partBeat + cursors.get(voice);
        const duration = elementDurationBeats(element);
        const endBeat = startBeat + duration;
        const elementNotes = Array.isArray(element.notes) ? element.notes : [];
        for (const note of elementNotes) {
          if (isRest(element, note)) continue;
          const midi = note.pitch
            ? midiFromPitch(note.pitch)
            : degreeMidi(note.degree, fifths, accidentalStates.get(voice));
          if (midi === null) {
            warnings.push(`小节 ${measure.number || "?"} 存在无法转换的音高，已跳过`);
            continue;
          }
          notes.push({
            start: startBeat,
            end: endBeat,
            pitch: midi,
            track,
            voice,
            staff,
          });
        }
        cursors.set(voice, Math.max(cursors.get(voice), cursors.get(voice) + duration));
      }
      const measureBeats = Math.max(0, ...cursors.values());
      partBeat += measureBeats;
    }
    durationBeats = Math.max(durationBeats, partBeat);
  }
  if (!notes.length) throw new Error("jpeditor 没有输出有声音符");

  const bpm = Number(options.bpm || 100);
  const safeBpm = Number.isFinite(bpm) && bpm > 0 ? bpm : 100;
  const secondsPerBeat = 60 / safeBpm;
  const richNotes = notes
    .map((note) => ({
      start: note.start * secondsPerBeat,
      end: note.end * secondsPerBeat,
      pitch: note.pitch,
      track: note.track,
    }))
    .sort((a, b) => a.start - b.start || a.track - b.track || a.pitch - b.pitch);
  const duration = Math.max(
    durationBeats * secondsPerBeat,
    ...richNotes.map((note) => note.end),
  );
  if (options.mode === "handwritten") {
    warnings.push("当前 jpeditor 组件没有经过手写谱训练/验收，手写模式仅作实验性尝试，必须人工确认");
  }
  return {
    format: "auto-music-player-source",
    version: 1,
    name: String(song.work?.title || "简谱 OMR 结果"),
    bpm: Math.round(safeBpm),
    notes: richNotes,
    tracks,
    duration,
    tempo_change_count: 1,
    recognizer: "jpeditor-omr",
    recognizer_version: String(options.recognizerVersion || "0.7.6"),
    mode: String(options.mode || "printed"),
    warnings,
    raw_text: String(options.rawText || ""),
    manual_confirmation_required: true,
  };
}

export { midiFromPitch, degreeMidi };
