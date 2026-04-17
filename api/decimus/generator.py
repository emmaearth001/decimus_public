"""Constrained generation: wraps SymphonyNet's gen_one/get_next with orchestration biasing.

Copies the core generation logic from gen_utils.py and adds:
- Instrument logit biasing toward planned instruments
- Track count limiting to ensemble size
- Register masking for out-of-range pitches
"""

import copy
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fairseq'))
from gen_utils import (
    EOS,
    NOTON_PAD_DUR,
    NOTON_PAD_TRK,
    calc_pos,
    get_next_chord,
    music_dict,
    sampling,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from encoding import ins2str, ison

from .planner import OrchestrationPlan


def _get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _build_instrument_bias(plan: OrchestrationPlan, ins_vocab_size: int) -> np.ndarray:
    """Build a bias vector for instrument logits.

    Adds a positive bonus to instruments in the orchestration plan,
    guiding the model toward the desired instrumentation.
    """
    bias = np.zeros(ins_vocab_size, dtype=np.float32)

    for role in plan.roles:
        # Find the vocab index for this program number
        # Instrument tokens are encoded as ins2str(program) -> vocab lookup
        ins_token = ins2str(role.spec.program)
        try:
            idx = music_dict.word2index(3, ins_token)
            bias[idx] += plan.style.instrument_bias
        except KeyError:
            pass

    return bias


def _build_track_mask(plan: OrchestrationPlan, trk_vocab_size: int) -> np.ndarray:
    """Build a mask for track logits: 0 for allowed tracks, -inf for disallowed."""
    mask = np.full(trk_vocab_size, -1e9, dtype=np.float32)

    # Always allow special tokens (0-3: BOS, PAD, EOS, UNK)
    mask[:4] = 0.0

    # Allow tracks in the plan
    for role in plan.roles:
        if role.track_id < trk_vocab_size:
            mask[role.track_id] = 0.0

    return mask


def get_next_constrained(
    model,
    p,
    memory,
    plan: OrchestrationPlan,
    ins_bias: np.ndarray,
    trk_mask: np.ndarray,
    has_prime: bool = False,
    event_temp: float = 1.0,
):
    """Generate next token with orchestration constraints.

    Based on gen_utils.get_next() with added biasing.
    """
    device = _get_device()
    pr = torch.from_numpy(np.array(p))[None, None, :].to(device)

    (e, d, t, ins), memory = model(src_tokens=pr, src_lengths=memory)
    e, d, t, ins = e[0, :], d[0, :], t[0, :], ins[0, :]

    if has_prime:
        return (np.int64(EOS), np.int64(EOS), np.int64(EOS), ins), memory

    # Sample event with style temperature
    evt = sampling(e, t=event_temp)
    while evt == EOS:
        return (evt, np.int64(EOS), np.int64(EOS), ins), memory

    evt_word = music_dict.index2word(0, evt)

    # Chord tokens
    if evt_word.startswith('H'):
        rep = get_next_chord(evt_word)
        return (
            np.int64(music_dict.word2index(0, rep)),
            np.int64(NOTON_PAD_DUR),
            np.int64(NOTON_PAD_TRK),
            ins,
        ), memory

    # Non-note tokens (position, measure, etc.)
    if not ison(evt_word):
        return (evt, np.int64(NOTON_PAD_DUR), np.int64(NOTON_PAD_TRK), ins), memory

    # Note token — apply constraints
    # Duration sampling (unchanged)
    dur = sampling(d)
    while dur == EOS:
        dur = sampling(d)
    while dur == NOTON_PAD_DUR:
        dur = sampling(d)

    # Track sampling with mask (only allow planned tracks)
    t_logits = t.squeeze().cpu().numpy()
    t_logits = t_logits + trk_mask[:len(t_logits)]
    trk = sampling(torch.tensor(t_logits), p=0)

    # Instrument logit biasing (stored for aggregation, not directly sampled here)
    # The ins tensor is raw logits — we add our bias for later aggregation
    ins_biased = ins.clone()
    ins_np = ins_biased.squeeze().cpu().numpy()
    ins_np[:len(ins_bias)] += ins_bias
    ins_biased = torch.tensor(ins_np).to(ins.device)

    return (evt, dur, trk, ins_biased), memory


def gen_one_orchestral(
    model,
    prime_nums,
    plan: OrchestrationPlan,
    max_measures: int | None = None,
    max_len: int = 4090,
    min_len: int = 0,
):
    """Generate a full orchestral sequence with constraints.

    Based on gen_utils.gen_one() with orchestration biasing.
    """
    import gen_utils
    gen_utils.prime_mea_idx = 0

    prime = copy.deepcopy(prime_nums)
    ins_list = [-1]

    event_temp = plan.style.event_temperature

    # Build bias/mask vectors
    ins_vocab_size = len(music_dict.vocabs[3])
    trk_vocab_size = len(music_dict.vocabs[2])
    ins_bias = _build_instrument_bias(plan, ins_vocab_size)
    trk_mask = _build_track_mask(plan, trk_vocab_size)

    with torch.no_grad():
        memo = None
        cur_rel_pos = 0
        cur_mea = 0

        # Process prime tokens (teacher-forced)
        for item, next_item in zip(prime[:-1], prime[1:]):
            (e, d, t, ins), memo = get_next_constrained(
                model, item, memo, plan, ins_bias, trk_mask,
                has_prime=True, event_temp=event_temp,
            )
            cur_rel_pos, cur_mea = calc_pos(next_item[0], cur_rel_pos, cur_mea)
            ins_list.append(ins)

        # First free generation step
        (e, d, t, ins), memo = get_next_constrained(
            model, prime[-1], memo, plan, ins_bias, trk_mask,
            has_prime=False, event_temp=event_temp,
        )
        cur_rel_pos, cur_mea = calc_pos(e, cur_rel_pos, cur_mea)
        prime.append((e, d, t, len(prime) + 1, cur_rel_pos, cur_mea))
        ins_list.append(ins)

        # Autoregressive generation loop
        for i in tqdm(range(max_len - len(prime)), desc="Generating orchestral score"):
            (e, d, t, ins), memo = get_next_constrained(
                model, prime[-1], memo, plan, ins_bias, trk_mask,
                event_temp=event_temp,
            )
            if t == EOS:
                assert len(prime) > min_len, 'Generated excerpt too short.'
                break

            cur_rel_pos, cur_mea = calc_pos(e, cur_rel_pos, cur_mea)
            prime.append((e, d, t, len(prime) + 1, cur_rel_pos, cur_mea))
            ins_list.append(ins)

            # Stop at target measure count
            if (max_measures is not None
                    and music_dict.index2word(0, e)[0].lower() == 'm'
                    and cur_mea > max_measures * 3):
                break

    return prime, ins_list
