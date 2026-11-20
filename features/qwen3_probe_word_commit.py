"""Commit the frozen first-class action only when the greedy word changes."""

from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass


class Qwen3BatchWordChangingCommit(Qwen3BatchProbeFirstClass):
    def _budget(self, source):
        rows, budget, details = super()._budget(source)
        self.commit_proposal_rows = tuple(rows)
        return rows, budget, details

    def _finish_decoder(self, module, args, output):
        self.commit_proposal_rows = ()
        if self.diagnostic_zero_replay:
            return super()._finish_decoder(module, args, output)
        proposed_tokens = None

        def remember_argmax(_module, _args, logits):
            nonlocal proposed_tokens
            proposed_tokens = logits.detach().argmax(-1).reshape(-1)

        # The inherited acceptance test already computes the corrected logits.
        # Observe that readout rather than issuing another LM-head call.
        handle = self.head.register_forward_hook(remember_argmax)
        try:
            result = super()._finish_decoder(module, args, output)
        finally:
            handle.remove()
        if not self.commit_proposal_rows:
            return result
        assert proposed_tokens is not None and len(proposed_tokens) == len(self.active)
        rejected_rows, rejected_positions = [], []
        for position, row in enumerate(self.commit_proposal_rows):
            event = self.events[row][-1]
            assert event['step'] == self.step
            event['probability_decrease_accepted'] = event['accepted']
            event['proposed_token'] = int(proposed_tokens[row])
            reject = event['accepted'] and event['proposed_token'] == event['token']
            event['rejected_no_token_change'] = reject
            if reject:
                event['accepted'] = False
                result.last_hidden_state[row, -1] = self.reference_hidden[row, 0]
                rejected_rows.append(row)
                rejected_positions.append(position)
        if rejected_rows:
            cache = self.tail_kwargs['past_key_values']
            for layer, (keys, values) in self.cache_backup.items():
                cache.layers[layer].keys[rejected_rows, :, -1:] = keys[rejected_positions]
                cache.layers[layer].values[rejected_rows, :, -1:] = values[rejected_positions]
        return result

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            events = [e for e in self.events.get(row, ()) if e['step'] < len(tokens) and 'accepted' in e]
            stats.update(
                acceptance_requires_token_change=True,
                rejected_unchanged_token_calls=sum(e.get('rejected_no_token_change', False) for e in events),
                probability_decrease_proposals=sum(e['actual_log_probability_change'] < 0 for e in events),
                additional_acceptance_lm_head_calls=0,
            )
        return result
