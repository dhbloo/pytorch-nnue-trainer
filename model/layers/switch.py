"""Switchable mixture-of-experts layers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activation import build_activation_layer
from utils.quant_utils import fake_quant


class SwitchGate(nn.Module):
    """Switch Gating for MoE networks"""

    def __init__(self, num_experts: int, jitter_eps=0.0, no_scaling=False) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.jitter_eps = jitter_eps
        self.no_scaling = no_scaling

    def forward(self, route_logits: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """
        Apply switch gating to routing logits.
        Args:
            route_logits: float tensor of (batch_size, num_experts).
        Returns:
            route_idx: routing index, long tensor of (batch_size,).
            route_multiplier: multipler for expert outputs, float tensor of (batch_size,).
            load_balancing_loss: load balancing loss, float tensor of ().
            aux_outputs: auxiliary outputs, dict.
        """
        # add random jittering when training
        if self.training and self.jitter_eps > 0:
            noise = torch.rand_like(route_logits)
            noise = noise * 2 * self.jitter_eps + 1 - self.jitter_eps
            route_logits = route_logits * noise

        # get routing probabilities and index
        route_probs = torch.softmax(route_logits, dim=1)  # [B, num_experts]
        route_prob_max, route_idx = torch.max(route_probs, dim=1)  # [B]

        # calc load balancing loss
        inv_batch_size = 1.0 / route_logits.shape[0]
        route_frac = torch.tensor(
            [(route_idx == i).sum() * inv_batch_size for i in range(self.num_experts)],
            dtype=route_probs.dtype,
            device=route_probs.device,
        )  # [num_experts]
        route_prob_mean = route_probs.mean(0)  # [num_experts]
        load_balancing_loss = self.num_experts * torch.dot(route_frac, route_prob_mean)
        load_balancing_loss = load_balancing_loss - 1.0

        if self.no_scaling:
            route_multiplier = route_prob_max / route_prob_max.detach()
        else:
            route_multiplier = route_prob_max

        aux_outputs = {
            "route_prob_max": route_prob_max,
            "route_frac_min": route_frac.min(),
            "route_frac_max": route_frac.max(),
            "route_frac_std": route_frac.std(),
        }
        return route_idx, route_multiplier, load_balancing_loss, aux_outputs


class SwitchLinear(nn.Module):
    """Switchable linear layer for MoE networks"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert num_experts > 0, "Number of experts must be at least 1"
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty((num_experts, out_features * in_features), **factory_kwargs))
        self.weight_fact = nn.Parameter(torch.empty((1, out_features * in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts, out_features), **factory_kwargs))
            self.bias_fact = nn.Parameter(torch.empty((1, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # We initialize the weights and biases to be uniform on all experts
        weight_fact = self.weight_fact.view(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(weight_fact, a=math.sqrt(5))
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_fact)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_fact, -bound, bound)
            nn.init.zeros_(self.bias)

    def get_weight_and_bias(self, route_index: Tensor) -> tuple[Tensor, None | Tensor]:
        batch_size = route_index.shape[0]
        expert_weight = F.embedding(route_index, self.weight) + self.weight_fact
        expert_weight = expert_weight.view(batch_size, self.out_features, self.in_features)
        if self.bias is not None:
            expert_bias = F.embedding(route_index, self.bias) + self.bias_fact
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def get_weight_and_bias_at_idx(self, index: int) -> tuple[Tensor, None | Tensor]:
        expert_weight = self.weight[index] + self.weight_fact[0]
        expert_weight = expert_weight.view(self.out_features, self.in_features)
        if self.bias is not None:
            expert_bias = self.bias[index] + self.bias_fact[0]
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def forward(self, input: Tensor, route_index: Tensor) -> Tensor:
        expert_weight, expert_bias = self.get_weight_and_bias(route_index)
        output = torch.einsum("bmn,bn->bm", expert_weight, input)
        if expert_bias is not None:
            output = output + expert_bias
        return output


class SwitchLinearBlock(nn.Module):
    """LinearBlock with switchable linear layer for MoE networks"""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_experts,
        activation="relu",
        bias=True,
        quant=False,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        bias_quant_bits=32,
    ) -> None:
        super().__init__()
        self.fc = SwitchLinear(in_dim, out_dim, num_experts, bias)
        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_bits = bias_quant_bits

        # initialize activation
        self.activation = build_activation_layer(activation)

    def forward(self, x: Tensor, route_index: Tensor):
        if self.quant:
            weight, bias = self.fc.get_weight_and_bias(route_index)
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if bias is not None:
                b = fake_quant(
                    bias, self.weight_quant_scale * self.input_quant_scale, num_bits=self.bias_quant_bits
                )
                out = F.linear(x, w, b)
            else:
                out = F.linear(x, w)
        else:
            out = self.fc(x, route_index)

        if self.activation:
            out = self.activation(out)
        return out


class SwitchConv2d(nn.Module):
    """Switchable 2D convolution with per-expert weights for MoE networks."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int,
        kernel_size: int,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        from torch.nn.modules.utils import _pair, _reverse_repeat_tuple

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_experts = num_experts
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        num_weight_params = math.prod(self.weight_shape)
        self.weight = nn.Parameter(torch.empty((num_experts, num_weight_params), **factory_kwargs))
        self.weight_fact = nn.Parameter(torch.empty((1, num_weight_params), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts, out_channels), **factory_kwargs))
            self.bias_fact = nn.Parameter(torch.empty((1, out_channels), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # We initialize the weights and biases to be uniform on all experts
        weight_fact = self.weight_fact.view(self.weight_shape)
        nn.init.kaiming_uniform_(weight_fact, a=math.sqrt(5))
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_fact)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_fact, -bound, bound)
            nn.init.zeros_(self.bias)

    def get_weight_and_bias(self, route_index: Tensor) -> tuple[Tensor, None | Tensor]:
        batch_size = route_index.shape[0]
        expert_weight = F.embedding(route_index, self.weight) + self.weight_fact
        expert_weight = expert_weight.view(batch_size, *self.weight_shape)
        if self.bias is not None:
            expert_bias = F.embedding(route_index, self.bias) + self.bias_fact
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def get_weight_and_bias_at_idx(self, index: int) -> tuple[Tensor, None | Tensor]:
        expert_weight = (self.weight[index] + self.weight_fact[0]).view(self.weight_shape)
        if self.bias is not None:
            expert_bias = self.bias[index] + self.bias_fact[0]
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def forward(self, input: Tensor, route_index: Tensor):
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding

        batch_size = route_index.shape[0]
        expert_weight, expert_bias = self.get_weight_and_bias(route_index)
        expert_weight = expert_weight.reshape(-1, *self.weight_shape[1:])
        if expert_bias is not None:
            expert_bias = expert_bias.view(-1)

        input = input.view(1, -1, input.size(-2), input.size(-1))
        output = F.conv2d(
            input, expert_weight, expert_bias, self.stride, padding, self.dilation, self.groups * batch_size
        )
        output = output.view(batch_size, self.weight_shape[0], output.size(-2), output.size(-1))
        return output


class SwitchConv2dBlock(nn.Module):
    """Conv2dBlock with switchable convolution for MoE networks."""
    def __init__(
        self,
        in_dim,
        out_dim,
        num_experts,
        ks,
        st=1,
        padding=0,
        activation="relu",
        pad_type="zeros",
        bias=True,
        dilation=1,
        groups=1,
        activation_first=False,
        quant=False,
        input_quant_scale=128,
        input_quant_bits=8,
        weight_quant_scale=128,
        weight_quant_bits=8,
        bias_quant_scale=None,
        bias_quant_bits=32,
    ):
        super().__init__()
        assert pad_type in [
            "zeros",
            "reflect",
            "replicate",
            "circular",
        ], f"Unsupported padding mode: {pad_type}"
        self.activation_first = activation_first
        self.activation = build_activation_layer(activation)
        self.conv = SwitchConv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            num_experts=num_experts,
            kernel_size=ks,
            stride=st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
        )

        self.quant = quant
        if quant:
            assert self.conv.padding_mode == "zeros", "quant conv requires zero padding"
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def forward(self, x: Tensor, route_index: Tensor):
        if self.activation and self.activation_first:
            x = self.activation(x)
        if self.quant:
            batch_size = x.shape[0]
            weight, bias = self.conv.get_weight_and_bias(route_index)
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            x = x.view(1, -1, *x.shape[2:])
            w = fake_quant(weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            w = w.view(-1, *w.shape[2:])
            if bias is not None:
                bias = fake_quant(bias, self.bias_quant_scale, num_bits=self.bias_quant_bits)
                bias = bias.reshape(-1)
            x = F.conv2d(
                x,
                w,
                bias,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups * batch_size,
            )
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))
        else:
            x = self.conv(x, route_index)
        if not self.activation_first and self.activation:
            x = self.activation(x)
        return x
