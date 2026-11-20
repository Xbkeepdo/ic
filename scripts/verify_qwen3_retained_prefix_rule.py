"""A retained hidden update can preserve one token and change the next target."""

import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_qwen3_retained_prefix_state import verify_native_state_prefix


def verify():
    # Same greedy first token; lower original-target probability is accepted.
    native_memory, updated_memory = 2., 1.5
    probability = lambda memory: math.exp(memory) / (math.exp(memory) + math.exp(1.))
    assert native_memory > 1. and updated_memory > 1.
    assert probability(updated_memory) < probability(native_memory)
    # A later query reads the two different cached states, not just token0.
    native_next = 2 if 2 * native_memory - 3.1 > 0 else 1
    updated_next = 2 if 2 * updated_memory - 3.1 > 0 else 1
    assert native_next == 2 and updated_next == 1
    native, updated = [0, native_next], [0, updated_next]
    events = [dict(step=0, token=0, accepted=True), dict(step=1, token=updated_next, accepted=True)]
    assert verify_native_state_prefix(native, updated, events, events[0])
    assert not verify_native_state_prefix(native, updated, events, events[1])
    assert native[:1] == updated[:1] and native[1] != events[1]['token']
    # Rejected proposals do not justify changing either prefix or raw target.
    rejected = [dict(step=0, token=0, accepted=False), events[1]]
    for current, prior, event in ((updated, rejected, rejected[1]),
                                  ([3, 2], [dict(step=1, token=2, accepted=True)], dict(step=1, token=2, accepted=True))):
        try:
            verify_native_state_prefix(native, current, prior, event)
        except AssertionError:
            pass
        else:
            raise AssertionError('Changed native state before any retained action was missed.')
    return dict(passed=True, same_first_greedy_token=True, accepted_probability_decrease=True,
                different_later_target_with_identical_emitted_prefix=True,
                earlier_rejected_proposals_do_not_exempt_checks=True,
                both_prefix_and_target_checked_before_first_retained=True)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
