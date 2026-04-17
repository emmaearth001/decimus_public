"""Tests for output.py — MIDI writing and summary generation."""

import os

from decimus.output import summarize_output, write_orchestral_midi


class TestSummarizeOutput:
    def test_empty_sequence(self, romantic_plan):
        result = summarize_output([], romantic_plan)
        assert result["total_notes"] == 0
        assert result["num_tracks"] == 0
        assert result["duration_32nds"] == 0

    def test_counts_notes_per_track(self, romantic_plan):
        # Create some fake note tuples: (pitch, program, start_32nd, end_32nd, track_id)
        tid = romantic_plan.roles[0].track_id
        note_seq = [
            (60, 40, 0, 16, tid),
            (64, 40, 16, 32, tid),
            (67, 40, 32, 48, tid),
        ]
        result = summarize_output(note_seq, romantic_plan)
        assert result["total_notes"] == 3
        assert result["num_tracks"] == 1

    def test_duration_from_max_end(self, romantic_plan):
        tid = romantic_plan.roles[0].track_id
        note_seq = [
            (60, 40, 0, 100, tid),
            (64, 40, 50, 200, tid),
        ]
        result = summarize_output(note_seq, romantic_plan)
        assert result["duration_32nds"] == 200


class TestWriteOrchestralMidi:
    def test_creates_file(self, romantic_plan, tmp_midi_path):
        tid = romantic_plan.roles[0].track_id
        prog = romantic_plan.roles[0].spec.program
        note_seq = [
            (60, prog, 0, 16, tid, 90),
            (64, prog, 16, 32, tid, 85),
        ]
        path = write_orchestral_midi(note_seq, romantic_plan, tmp_midi_path)
        assert os.path.exists(path)

    def test_empty_sequence_creates_file(self, romantic_plan, tmp_midi_path):
        path = write_orchestral_midi([], romantic_plan, tmp_midi_path)
        assert os.path.exists(path)

    def test_velocity_default(self, romantic_plan, tmp_midi_path):
        """Notes without velocity (5-tuple) should default to 90."""
        tid = romantic_plan.roles[0].track_id
        prog = romantic_plan.roles[0].spec.program
        note_seq = [
            (60, prog, 0, 16, tid),  # no velocity field
        ]
        # Should not crash
        path = write_orchestral_midi(note_seq, romantic_plan, tmp_midi_path)
        assert os.path.exists(path)
