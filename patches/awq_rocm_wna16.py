"""
Silence AWQMoeMarlin warnings on ROCm.

On ROCm, check_moe_marlin_supports_layer() always returns False, so every
MoE layer triggers a 'not supported by AWQMoeMarlin, falling back to WNA16'
warning. On ROCm this is expected and correct — Marlin is CUDA-only. This
patch skips the Marlin check entirely on ROCm, sending all MoE layers
directly to MoeWNA16 without the noise.
"""
from vllm.platforms import current_platform


def apply():
    if not current_platform.is_rocm():
        return  # Only needed on ROCm

    import torch
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.awq import AWQConfig

    original_get_quant_method = AWQConfig.get_quant_method

    def patched_get_quant_method(self, layer, prefix):
        if isinstance(layer, FusedMoE):
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
            config = {
                "quant_method": "awq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "zero_point": self.zero_point,
                "lm_head": False,
                "modules_to_not_convert": self.modules_to_not_convert,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)
        return original_get_quant_method(self, layer, prefix)

    AWQConfig.get_quant_method = patched_get_quant_method
    print("[patch] AWQConfig: routing all MoE layers directly to WNA16 on ROCm (no Marlin)")
