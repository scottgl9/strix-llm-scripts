"""
Post-load quantization of lm_head weights to FP8 (e4m3fnuz for ROCm).

The lm_head is a ParallelLMHead with an unquantized weight of shape
(248320, 3072) in FP16/BF16 (~1.46 GB). This patch quantizes it to FP8
after weight loading, halving the memory footprint and reducing memory
bandwidth pressure during the vocab projection.

The approach: monkey-patch the model's load_weights to run FP8 conversion
on lm_head.weight after the original load completes. We replace the
quant_method.apply to handle FP8 dequant-matmul.
"""
import torch
import torch.nn.functional as F


def _quantize_weight_fp8(weight):
    """Quantize a weight tensor to FP8 e4m3fnuz with per-tensor scale."""
    fp8_dtype = torch.float8_e4m3fnuz
    fp8_max = torch.finfo(fp8_dtype).max
    w_float = weight.float()
    amax = w_float.abs().max().clamp(min=1e-12)
    scale = amax / fp8_max
    w_fp8 = (w_float / scale).to(fp8_dtype)
    return w_fp8, scale


def _make_fp8_apply(original_apply):
    """Create an apply method that handles FP8 weights."""
    def fp8_apply(self, layer, x, bias=None):
        if hasattr(layer, '_fp8_scale'):
            # Dequantize FP8 weight to input dtype and matmul
            w = layer.weight.to(x.dtype) * layer._fp8_scale.to(x.dtype)
            return F.linear(x, w, bias)
        else:
            return original_apply(self, layer, x, bias=bias)
    return fp8_apply


def apply():
    """Patch Qwen3_5MoeForCausalLM.load_weights to quantize lm_head post-load."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    original_load_weights = Qwen3_5ForCausalLMBase.load_weights

    def patched_load_weights(self, weights):
        result = original_load_weights(self, weights)

        # Quantize lm_head if it exists and has a weight
        if hasattr(self, 'lm_head') and hasattr(self.lm_head, 'weight'):
            w = self.lm_head.weight
            if w.dtype != torch.float8_e4m3fnuz:
                old_bytes = w.numel() * w.element_size()
                w_fp8, scale = _quantize_weight_fp8(w.data)
                self.lm_head.weight = torch.nn.Parameter(w_fp8, requires_grad=False)
                self.lm_head._fp8_scale = scale
                new_bytes = w_fp8.numel() * w_fp8.element_size()
                print(f"[quantize_lm_head] lm_head quantized to FP8: "
                      f"{old_bytes/1024**2:.0f}MB -> {new_bytes/1024**2:.0f}MB")

                # Patch the quant_method.apply to handle FP8
                qm = self.lm_head.quant_method
                original_apply = type(qm).apply
                type(qm).apply = _make_fp8_apply(original_apply)

                torch.cuda.empty_cache()

        return result

    Qwen3_5ForCausalLMBase.load_weights = patched_load_weights
    print("[quantize_lm_head] Patched load_weights to quantize lm_head to FP8")
