"""Add a selective FFN adjustment to visually grounded attention feedback."""

import torch

from features.qwen3_sar_ems_alternative import Qwen3BatchSupportedAlternative


class Qwen3BatchAlternativeWithFFNAux(Qwen3BatchSupportedAlternative):
    """Remove part of the FFN's gain in the unsupported primary-rival margin."""

    def _after_layer(self, source):
        def finish(_module, _args, output):
            rows, budget, details = self._budget(source)
            state = self.scratch.pop(source)
            if not rows:
                return output
            self.pending[source + 1] = (source, rows, budget, details)
            residual, update = state['ffn_residual'], state['update']
            norm = self.decoder.norm
            after = norm(residual + update)
            primary = self.head(after).argmax(-1)[rows]
            before = norm(residual)[rows]
            rival = primary.new_tensor([detail['token'] for detail in details])
            weights = (self.head.weight[primary].float() - self.head.weight[rival].float())
            gain = (weights * (after[rows].float() - before.float())).sum(-1).clamp_min(0)

            # Remove at most half of the added primary-rival margin under
            # a local final-norm logit approximation. Bound the residual
            # adjustment to 15% of the FFN update's norm.
            h = output[rows, -1].float()
            variance = h.square().mean(-1, keepdim=True) + norm.variance_epsilon
            direction = weights * norm.weight.float()
            direction = (direction - h * (h * direction).mean(-1, keepdim=True) / variance) / variance.sqrt()
            scale = 0.5 * gain / direction.square().sum(-1).clamp_min(1e-12)
            correction = scale[:, None] * direction
            limit = 0.15 * update[rows].float().norm(dim=-1, keepdim=True)
            correction = correction * (limit / correction.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1)
            edited = output.clone()
            edited[rows, -1] = (h - correction).to(output.dtype)
            return edited

        return finish
