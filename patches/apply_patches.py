#!/usr/bin/env python3
"""Apply all vLLM patches, then launch vllm serve with the original arguments."""
import sys

import rmsnorm_gated_activation
rmsnorm_gated_activation.apply()

if __name__ == "__main__":
    from vllm.scripts import main
    sys.exit(main())
