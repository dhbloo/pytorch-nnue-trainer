import torch
import torch.nn as nn
from . import MODELS


@MODELS.register('linear')
class LinearModel(nn.Module):
    def __init__(self,
                 pcode_dim=2380,
                 has_draw_value=True,
                 two_side=False,
                 fixed_side_input=False) -> None:
        super().__init__()
        self.pcode_dim = pcode_dim
        self.has_draw_value = has_draw_value
        self.two_side = two_side
        self.fixed_side_input = fixed_side_input
        self.table = nn.Embedding(num_embeddings=(2 if two_side else 1) * pcode_dim,
                                  embedding_dim=2 + (1 if self.has_draw_value else 0))
        self.policy_stm_coef = nn.parameter.Parameter(torch.ones(2), True)

    def forward(self, data):
        assert torch.all(self.pcode_dim == data['sparse_feature_dim'][:, 10:12])
        pcode_sparse_input = data['sparse_feature_input'][:, [10, 11]].int()  # [B, 2, H, W]
        if self.two_side:
            # add index offset for opponent side
            pcode_sparse_input[:, 1] += self.pcode_dim

        # lookup value and policy using sparse input
        map = self.table(pcode_sparse_input)  # [B, 2, H, W, 2 or 3]
        policy_map = map[..., 0].contiguous()  # [B, 2, H, W]
        value_map = map[..., 1:].contiguous()  # [B, 2, H, W, 1 or 2]

        # mask out value and policy at non-empty cell
        board_input = data['board_input']  # [B, 2, H, W]
        non_empty_mask = board_input[:, 0] + board_input[:, 1] > 0  # [B, H, W]
        non_empty_mask = torch.unsqueeze(non_empty_mask, dim=1)  # [B, 1, H, W]
        value_map = torch.masked_fill(value_map, non_empty_mask.unsqueeze(4),
                                      0)  # [B, 2, H, W, 1 or 2]
        policy_map = torch.masked_fill(policy_map, non_empty_mask, 0)  # [B, 2, H, W]

        # combine two side's value and policy
        value = torch.sum(value_map, dim=(2, 3))  # [B, 2, 1 or 2]
        if self.has_draw_value:
            win = value[:, 0, 0] - value[:, 1, 0]  # [B]
            loss = -win  # [B]
            draw = value[:, 0, 1] + value[:, 1, 1]  # [B]
            value = torch.stack((win, loss, draw), dim=1)
        else:
            value = value[:, 0] - value[:, 1]  # [B, 1] no draw value

        if self.fixed_side_input:
            flip = data['stm_input']  # Black = -1, White = 1
            flip_cond = flip < 0

            if self.has_draw_value:
                win = torch.where(flip_cond, value[:, 0], value[:, 1])
                loss = torch.where(flip_cond, value[:, 1], value[:, 0])
                draw = value[:, 2]
                value = torch.stack((win, loss, draw), dim=1)
            else:
                value = value * torch.unsqueeze(flip * -1, dim=1)

            policy_self = torch.where(flip_cond[:, None, None], policy_map[:, 0],
                                      policy_map[:, 1])  # [B, H, W]
            policy_oppo = torch.where(flip_cond[:, None, None], policy_map[:, 1],
                                      policy_map[:, 0])  # [B, H, W]
        else:
            policy_self = policy_map[:, 0]
            policy_oppo = policy_map[:, 1]
        policy_self = policy_self * self.policy_stm_coef[0]
        policy_oppo = policy_oppo * self.policy_stm_coef[1]
        policy = policy_self + policy_oppo

        return value, policy

    @property
    def name(self):
        flags = 'draw' if self.has_draw_value else 'nodraw'
        if self.two_side: flags += '-twoside'
        if self.fixed_side_input: flags += '-fixed'
        return f"linear_{flags}"