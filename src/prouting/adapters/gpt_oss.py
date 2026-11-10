"""gpt-oss stub (optional model, PROJECT.md 8). Not implemented: its experts have biases, a clamped SwiGLU
variant, MXFP4 weights and softmax after top-k (verify). Implement only if time allows."""
from .base import FusedExpertsAdapter, UnknownLayoutError


class GptOssAdapter(FusedExpertsAdapter):
    BLOCK_CLASS = "GptOssMLP"
    FAMILY = "gpt_oss"

    def check_layout(self, block):
        raise UnknownLayoutError("gpt-oss adapter is a stub (not implemented)")
