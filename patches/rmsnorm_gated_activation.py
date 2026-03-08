"""
Patch for vLLM's RMSNormGated (CustomOp) class in layernorm.py.

Bug: The CustomOp version of RMSNormGated.__init__ does not set self.activation,
but forward_cuda references it when calling rmsnorm_fn(activation=self.activation).
The nn.Module version of RMSNormGated (later in the same file) correctly sets it.

Fix: Add self.activation = "swish" to the CustomOp RMSNormGated.__init__.
"""
import importlib
import sys


def apply():
    """Monkey-patch RMSNormGated(CustomOp) to add missing self.activation."""
    from vllm.model_executor.layers.layernorm import RMSNormGated

    original_init = RMSNormGated.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "activation"):
            self.activation = "swish"

    RMSNormGated.__init__ = patched_init
    print("[patch] RMSNormGated: added missing 'activation' attribute")


if __name__ == "__main__":
    apply()
