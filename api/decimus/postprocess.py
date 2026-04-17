"""Post-processing: validate instrument ranges, apply dynamics, clean output."""

from .instruments import clamp_to_range, in_range
from .planner import OrchestrationPlan


def postprocess_notes(
    note_seq: list[tuple],
    plan: OrchestrationPlan,
) -> list[tuple]:
    """Post-process generated note sequence.

    Each note is (pitch, program, start, end, track_id).
    Returns cleaned note sequence with:
    - Out-of-range notes transposed or removed
    - Velocity adjustments based on instrument role
    """
    cleaned = []
    transposed_count = 0
    removed_count = 0

    # Build track->role mapping for velocity
    track_velocity = {}
    for role in plan.roles:
        track_velocity[role.track_id] = int(90 * role.velocity_scale)

    for pitch, program, start, end, track_id in note_seq:
        # Find the instrument spec for this track
        spec = plan.track_to_instrument.get(track_id)
        if spec is None:
            # Unknown track — keep note as-is
            cleaned.append((pitch, program, start, end, track_id))
            continue

        # Validate range
        if not in_range(pitch, spec):
            new_pitch = clamp_to_range(pitch, spec)
            if in_range(new_pitch, spec):
                pitch = new_pitch
                transposed_count += 1
            else:
                removed_count += 1
                continue

        cleaned.append((pitch, program, start, end, track_id))

    return cleaned


def apply_dynamics(
    note_seq: list[tuple],
    plan: OrchestrationPlan,
) -> list[tuple]:
    """Apply dynamic velocity mapping based on role.

    Returns notes as (pitch, program, start, end, track_id, velocity).
    """
    result = []
    for pitch, program, start, end, track_id in note_seq:
        # Default velocity
        velocity = 90

        # Look up role-based velocity
        for role in plan.roles:
            if role.track_id == track_id:
                velocity = int(90 * role.velocity_scale)
                break

        result.append((pitch, program, start, end, track_id, velocity))
    return result
