"""Keep the frozen head-gradient intervention eligible for repeated objects."""

from features.qwen3_sar_ems_head_gradient import Qwen3BatchHeadGradient


class Qwen3BatchRepeatedObjectGuidance(Qwen3BatchHeadGradient):
    def _budget(self, source):
        # Only the previous-token exclusion changes. Keep the true history for
        # the next decode step and for separating first and repeated proposals.
        history = self.seen
        self.seen = [set() for _ in history]
        try:
            rows, budget, details = super()._budget(source)
        finally:
            self.seen = history
        for row, detail in zip(rows, details, strict=True):
            detail['previously_generated_token'] = detail['token'] in history[row]
        return rows, budget, details

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            events = [e for e in self.events.get(row, ())
                      if e['step'] < len(tokens) and e.get('previously_generated_token', False)]
            stats.update(
                repeated_objects_eligible=True,
                repeated_object_proposals=len(events),
                repeated_object_accepted=sum(e.get('accepted', False) for e in events),
                repeated_object_token_changes=sum(tokens[e['step']] != e['token'] for e in events),
            )
        return result
