import torch
import torch.nn as nn

from .blocks import Conv2dBlock, LinearBlock


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

    def forward(self, feature):
        # value head
        value = torch.mean(feature, dim=(2, 3))
        value = self.value_head(value)

        # policy head
        policy = self.policy_head(feature)
        policy = torch.squeeze(policy, dim=1)

        return value, policy
