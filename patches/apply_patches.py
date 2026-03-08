#!/usr/bin/env python3
"""Apply all vLLM patches, then launch vllm serve with the original arguments."""
import sys

import rmsnorm_gated_activation
rmsnorm_gated_activation.apply()

import awq_rocm_wna16
awq_rocm_wna16.apply()

# NOTE: FP8 lm_head quantization disabled — gfx1151 lacks native FP8 matmul
# (torch._scaled_mm requires MI300+ or SM89+), so the dequant overhead
# actually reduces throughput vs keeping lm_head in FP16.

if __name__ == "__main__":
    from vllm.scripts import main
    sys.exit(main())
