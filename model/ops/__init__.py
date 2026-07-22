"""Performance-sensitive model operators with reference PyTorch implementations.

Keep composite operators expressed in regular PyTorch here so TorchInductor can
inline and fuse them. Backend-specific implementations expose portable guards
and fallbacks without coupling model definitions to a single device.
"""

from .aggregation import (
    mean_3x3_regions,
    quantized_avg4,
    quantized_sum_3x3_regions,
    split_3x3_regions,
    sum_3x3_regions,
)
from .dynamic import dynamic_pointwise_conv2d
from .embedding import small_table_embedding
from .quantized import (
    pixelwise_quantized_depthwise_conv2d,
    product_quantization_has_linear_gradient,
)
from .vector_quantization import (
    accelerated_cosine_argmin,
    accelerated_l2_argmin,
    accelerated_perplexity_stats,
    supports_accelerated_cosine_argmin,
    supports_accelerated_l2_argmin,
)

__all__ = [
    "accelerated_cosine_argmin",
    "accelerated_l2_argmin",
    "accelerated_perplexity_stats",
    "dynamic_pointwise_conv2d",
    "small_table_embedding",
    "pixelwise_quantized_depthwise_conv2d",
    "product_quantization_has_linear_gradient",
    "mean_3x3_regions",
    "quantized_avg4",
    "quantized_sum_3x3_regions",
    "split_3x3_regions",
    "sum_3x3_regions",
    "supports_accelerated_cosine_argmin",
    "supports_accelerated_l2_argmin",
]
