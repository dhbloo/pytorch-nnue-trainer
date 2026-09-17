import torch
import torch.nn as nn

from .layers.convolution import Conv2dBlock
from .layers.linear import LinearBlock


def build_head(head_type, dim_feature):
    if "-nodraw" in head_type:
        dim_value = 1
        head_type = head_type.replace("-nodraw", "")
    else:
        dim_value = 3

    if head_type == "v0":
        return OutputHeadV0(dim_feature, dim_value)
    else:
        raise ValueError(f"Unsupported head: {head_type}")


class OutputHeadV0(nn.Module):
    def __init__(self, dim_feature, dim_value=3):
        super().__init__()
        self.value_head = LinearBlock(dim_feature, dim_value, activation="none", bias=False)
        self.policy_head = Conv2dBlock(dim_feature, 1, ks=1, st=1, activation="none", bias=False)

    def forward(self, feature: torch.Tensor, mask: None | torch.Tensor = None):
        # Large shared logits can erase class differences in reduced precision.
        # Keep pooling and the small value classifier in FP32, including when
        # the convolutional trunk runs under autocast or has half-type weights.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            if mask is not None:
                mask = mask.float()
                mask_sum = torch.sum(mask, dim=(2, 3), keepdim=False)
                value = torch.sum(feature.float() * mask, dim=(2, 3)) / mask_sum
            else:
                value = torch.mean(feature, dim=(2, 3), dtype=torch.float32)
            # This small dot product also avoids TF32 GEMM downcasting when
            # the global float32 matmul policy favors tensor-core throughput.
            value = (value.unsqueeze(1) * self.value_head.fc.weight.float()).sum(dim=-1)

        # policy head
        policy = self.policy_head(feature)
        policy = torch.squeeze(policy, dim=1)

        if mask is not None:
            mask = torch.squeeze(mask, dim=1)
            return {"value": value, "policy": policy, "board_mask": mask}
        else:
            return {"value": value, "policy": policy}
