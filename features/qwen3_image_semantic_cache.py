"""Cache original per-image E_l once, without reading decoding attention."""

from contextlib import ExitStack, contextmanager

from features.qwen3_online_i_el import OnlineIElCapture


class ImageSemanticCache(OnlineIElCapture):
    @contextmanager
    def capture(self):
        with ExitStack() as stack:
            handle = self.model.register_forward_pre_hook(self._begin, with_kwargs=True)
            stack.callback(handle.remove)
            for index, layer in enumerate(self.decoder.layers):
                handle = layer.register_forward_pre_hook(self._input(index), with_kwargs=True)
                stack.callback(handle.remove)
            yield self
