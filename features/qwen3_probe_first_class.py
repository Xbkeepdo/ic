"""Gate the frozen layer19 recovery before the first known object class."""

from features.qwen3_probe_early_recycle import Qwen3BatchProbeEarlyRecycle


class Qwen3BatchProbeFirstClass(Qwen3BatchProbeEarlyRecycle):
    def __init__(self, *args, tokenizer, history_parser, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.history_parser = history_parser
        self.token_classes = history_parser.token_classes(tokenizer, self.nouns)
        self.output_history = []
        self.class_decisions = {}

    def _begin_step(self, module, args, kwargs):
        continuation = bool(self.steps)
        super()._begin_step(module, args, kwargs)
        ids = kwargs.get('input_ids', args[0] if args else None)
        if not continuation:
            self.output_history = [[] for _ in range(len(ids))]
        else:
            for history, token in zip(self.output_history, ids[:, -1].tolist(), strict=True):
                history.append(token)

    def _budget(self, source):
        original_active = self.active
        self.active = self.active.clone()
        for row, (token, active) in enumerate(zip(self.final_tokens.tolist(), original_active.tolist(), strict=True)):
            if not active or token not in self.nouns or token in self.seen[row]:
                continue
            assert len(self.output_history[row]) == self.step
            text = self.tokenizer.decode(self.output_history[row], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
            prior = self.history_parser.classes(text)
            canonical = self.token_classes[token]
            blocked = canonical in prior
            self.class_decisions.setdefault(row, []).append(dict(step=self.step, token=token,
                canonical=canonical, prior_classes=sorted(prior), blocked=blocked))
            if blocked and not self.force_probe_gate:
                self.active[row] = False
        try:
            return super()._budget(source)
        finally:
            self.active = original_active

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            decisions = [d for d in self.class_decisions.get(row, ()) if d['step'] < len(tokens)]
            stats.update(first_object_class_gate=True, history_uses_generated_prefix_only=True,
                         class_decisions=decisions, blocked_by_class_history=sum(d['blocked'] for d in decisions))
        return result
