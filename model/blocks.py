import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant_utils import fake_quant


def build_activation_layer(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif activation.startswith('lrelu/'):  # custom slope
        neg_slope = 1.0 / int(activation[6:])
        return nn.LeakyReLU(neg_slope, inplace=True)
    elif activation == 'crelu':
        return ClippedReLU(inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU(inplace=True)
    elif activation == 'mish':
        return nn.Mish(inplace=True)
    elif activation == 'none':
        return None
    else:
        assert 0, f"Unsupported activation: {activation}"


def build_norm1d_layer(norm, norm_dim=None):
    if norm == 'bn':
        return nn.BatchNorm1d(norm_dim)
    elif norm == 'in':
        return nn.InstanceNorm1d(norm_dim)
    elif norm == 'none':
        return None
    else:
        assert 0, f"Unsupported normalization: {norm}"


def build_norm2d_layer(norm, norm_dim=None):
    if norm == 'bn':
        assert norm_dim is not None
        return nn.BatchNorm2d(norm_dim)
    elif norm == 'in':
        assert norm_dim is not None
        return nn.InstanceNorm2d(norm_dim)
    elif norm == 'none':
        return None
    else:
        assert 0, f"Unsupported normalization: {norm}"


# ---------------------------------------------------------
# Activation Layers


class ClippedReLU(nn.Module):
    def __init__(self, inplace=False, max=1):
        super().__init__()
        self.inplace = inplace
        self.max = max

    def forward(self, x: torch.Tensor):
        if self.inplace:
            return x.clamp_(0, self.max)
        else:
            return x.clamp(0, self.max)


class ChannelWiseLeakyReLU(nn.Module):
    def __init__(self, dim, bias=True, bound=0):
        super().__init__()
        self.neg_slope = nn.Parameter(torch.ones(dim) * 0.5)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.bound = bound

    def forward(self, x):
        assert x.ndim >= 2
        shape = [1, -1] + [1] * (x.ndim - 2)

        slope = self.neg_slope.view(shape)
        # limit slope to range [-bound, bound]
        if self.bound != 0:
            slope = torch.tanh(slope / self.bound) * self.bound

        x += -torch.relu(-x) * (slope - 1)
        if self.bias is not None:
            x += self.bias.view(shape)
        return x


class QuantPReLU(nn.PReLU):
    def __init__(self,
                 num_parameters: int = 1,
                 init: float = 0.25,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 weight_signed=True):
        super().__init__(num_parameters, init)
        self.input_quant_scale = input_quant_scale
        self.input_quant_bits = input_quant_bits
        self.weight_quant_scale = weight_quant_scale
        self.weight_quant_bits = weight_quant_bits
        self.weight_signed = weight_signed

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = fake_quant(
            input,
            self.input_quant_scale,
            num_bits=self.input_quant_bits,
        )
        weight = fake_quant(
            self.weight,
            self.weight_quant_scale,
            num_bits=self.weight_quant_bits,
            signed=self.weight_signed,
        )
        return F.prelu(input, weight)


class SwitchPReLU(nn.Module):
    def __init__(self,
                 num_parameters: int,
                 num_experts: int,
                 init: float = 0.25,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.zeros((num_experts, num_parameters), **factory_kwargs))
        self.weight_fact = nn.Parameter(torch.full((1, num_parameters), init, **factory_kwargs))

    def get_weight(self, route_index: torch.Tensor) -> torch.Tensor:
        return F.embedding(route_index, self.weight) + self.weight_fact

    def get_weight_at_idx(self, index: int) -> torch.Tensor:
        return self.weight[index] + self.weight_fact[0]

    def forward(self, input: torch.Tensor, route_index: torch.Tensor) -> torch.Tensor:
        neg_slope = self.get_weight(route_index)
        neg_slope = neg_slope.view(neg_slope.shape[0], neg_slope.shape[1],
                                   *([1] * (input.ndim - neg_slope.ndim)))
        return torch.where(input >= 0, input, neg_slope * input)


# ---------------------------------------------------------
# Neural Network Layers


class LinearBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm='none',
                 activation='relu',
                 bias=True,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_bits=32):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_bits = bias_quant_bits

        self.norm = build_norm1d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)

    def forward(self, x):
        if self.quant:
            # Using floor for inputs leads to closer results to the actual inference code
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits, floor=True)
            w = fake_quant(self.fc.weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if self.fc.bias is not None:
                b = fake_quant(self.fc.bias,
                               self.weight_quant_scale * self.input_quant_scale,
                               num_bits=self.bias_quant_bits)
                out = F.linear(x, w, b)
            else:
                out = F.linear(x, w)
        else:
            out = self.fc(x)

        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ks,
                 st,
                 padding=0,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
                 bias=True,
                 dilation=1,
                 groups=1,
                 activation_first=False,
                 use_spectral_norm=False,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_scale=None,
                 bias_quant_bits=32):
        super(Conv2dBlock, self).__init__()
        assert pad_type in ['zeros', 'reflect', 'replicate',
                            'circular'], f"Unsupported padding mode: {pad_type}"
        self.activation_first = activation_first
        self.norm = build_norm2d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)
        self.conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=ks,
            stride=st,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_type,
        )

        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.quant = quant
        if quant:
            assert self.conv.padding_mode == 'zeros', "quant conv requires zero padding"
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        if self.quant:
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(self.conv.weight,
                           self.weight_quant_scale,
                           num_bits=self.weight_quant_bits)
            b = self.conv.bias
            if b is not None:
                b = fake_quant(b, self.bias_quant_scale, num_bits=self.bias_quant_bits)
            if self.quant == 'pixel-dwconv':  # pixel-wise quantization in depthwise conv
                assert self.conv.groups == x.size(1), "must be dwconv in pixel-dwconv quant mode!"
                x_ = F.conv2d(x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
                x = F.unfold(x, self.conv.kernel_size, self.conv.dilation, self.conv.padding, self.conv.stride)
                x = fake_quant(x * w.view(-1)[None, :, None], self.bias_quant_scale, num_bits=self.bias_quant_bits)
                x = x.reshape(x_.shape[0], x_.shape[1], -1, x_.size(2) * x_.size(3)).sum(2)
                x = F.fold(x, (x_.size(2), x_.size(3)), (1, 1))
                x = x + b[None, :, None, None]
            else:
                x = F.conv2d(x, w, b, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        else:
            x = self.conv(x)
        if not self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class Conv1dLine4Block(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ks,
                 st,
                 padding=0,
                 norm='none',
                 activation='relu',
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
                 bias_quant_bits=32):
        super().__init__()
        assert in_dim % groups == 0, f"in_dim({in_dim}) should be divisible by groups({groups})"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = ks
        self.stride = st
        self.padding = padding  # zeros padding
        self.dilation = dilation
        self.groups = groups
        self.activation_first = activation_first
        self.norm = build_norm2d_layer(norm, out_dim)
        self.activation = build_activation_layer(activation)
        self.weight = nn.Parameter(torch.empty((4 * ks - 3, out_dim, in_dim // groups)))
        self.bias = nn.Parameter(torch.zeros((out_dim, ))) if bias else None
        nn.init.kaiming_normal_(self.weight)

        self.quant = quant
        if quant:
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def make_kernel(self):
        kernel = []
        weight_index = 0
        zero = torch.zeros_like(self.weight[0])
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == j or i + j == self.kernel_size - 1 or i == self.kernel_size // 2 or j == self.kernel_size // 2:
                    kernel.append(self.weight[weight_index])
                    weight_index += 1
                else:
                    kernel.append(zero)

        assert weight_index == self.weight.size(0), f"{weight_index} != {self.weight.size(0)}"
        kernel = torch.stack(kernel, dim=2)
        kernel = kernel.reshape(self.out_dim, self.in_dim // self.groups, self.kernel_size,
                                self.kernel_size)
        return kernel

    def conv(self, x):
        w = self.make_kernel()
        b = self.bias

        if self.quant:
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(w, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if b is not None:
                b = fake_quant(b, self.bias_quant_scale, num_bits=self.bias_quant_bits)

        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        x = self.conv(x)
        if not self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self,
                 dim_in,
                 ks=3,
                 st=1,
                 padding=1,
                 norm='none',
                 activation='relu',
                 pad_type='zeros',
                 dim_out=None,
                 dim_hidden=None,
                 activation_first=False):
        super(ResBlock, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = dim_hidden or min(dim_in, dim_out)

        self.learned_shortcut = dim_in != dim_out
        self.activation_first = activation_first
        self.activation = build_activation_layer(activation)
        self.conv = nn.Sequential(
            Conv2dBlock(dim_in,
                        dim_hidden,
                        ks,
                        st,
                        padding,
                        norm,
                        activation,
                        pad_type,
                        activation_first=activation_first),
            Conv2dBlock(dim_hidden,
                        dim_out,
                        ks,
                        st,
                        padding,
                        norm,
                        activation if activation_first else 'none',
                        pad_type,
                        activation_first=activation_first),
        )
        if self.learned_shortcut:
            self.conv_shortcut = Conv2dBlock(dim_in,
                                             dim_out,
                                             1,
                                             1,
                                             norm=norm,
                                             activation=activation,
                                             activation_first=activation_first)

    def forward(self, x):
        residual = self.conv_shortcut(x) if self.learned_shortcut else x
        out = self.conv(x)
        out += residual
        if not self.activation_first and self.activation:
            out = self.activation(out)
        return out


class HashLayer(nn.Module):
    """Maps an input space of size=(input_level**input_size) to a hashed feature space of size=(2**hash_logsize)."""
    def __init__(self,
                 input_size,
                 input_level=2,
                 hash_logsize=20,
                 dim_feature=32,
                 quant_int8=True,
                 sub_features=1,
                 sub_divisor=2,
                 scale_grad_by_freq=False):
        super().__init__()
        self.input_size = input_size
        self.input_level = input_level
        self.hash_logsize = hash_logsize
        self.dim_feature = dim_feature
        self.quant_int8 = quant_int8
        self.sub_features = sub_features
        self.sub_divisor = sub_divisor

        self.perfect_hash = 2**hash_logsize >= input_level**input_size
        if self.perfect_hash:  # Do perfect hashing
            n_features = input_level**input_size
            self.register_buffer('idx_stride',
                                 input_level**torch.arange(input_size, dtype=torch.int64))
        else:
            n_features = 2**hash_logsize
            ii32 = torch.iinfo(torch.int32)
            self.hashs = nn.Parameter(torch.randint(ii32.min,
                                                    ii32.max,
                                                    size=(input_size, input_level),
                                                    dtype=torch.int32),
                                      requires_grad=False)

        n_features = [math.ceil(n_features / sub_divisor**i) for i in range(sub_features)]
        self.offsets = [sum(n_features[:i]) for i in range(sub_features + 1)]
        level_dim = max(dim_feature // sub_features, 1)
        self.features = nn.Embedding(sum(n_features),
                                     level_dim,
                                     scale_grad_by_freq=scale_grad_by_freq)
        if self.quant_int8:
            nn.init.trunc_normal_(self.features.weight.data, std=0.5, a=-1, b=127 / 128)
        if self.sub_features > 1:
            self.feature_mapping = nn.Linear(level_dim * sub_features, dim_feature)

    def forward(self, x, x_long=None):
        """
        Args:
            x: float tensor of (batch_size, input_size), in range [0, 1].
            x_long: long tensor of (batch_size, input_size), in range [0, input_level-1].
                If not None, x_long will be used and x will be ignored.
        Returns:
            x_features: float tensor of (batch_size, dim_feature).
        """
        # Quantize input to [0, input_level-1] level
        if x_long is None:
            assert torch.all((x >= 0) & (x <= 1)), f"Input x should be in range [0, 1], but got {x}"
            x_long = torch.round((self.input_level - 1) * x).long()  # (batch_size, input_size)

        if self.perfect_hash:
            x_indices = torch.sum(x_long * self.idx_stride, dim=1)  # (batch_size,)
        else:
            x_onthot = torch.zeros((x_long.shape[0], self.input_size, self.input_level),
                                   dtype=torch.bool,
                                   device=x_long.device)
            x_onthot.scatter_(2, x_long.unsqueeze(-1), 1)  # (batch_size, input_size, input_level)
            x_hash = x_onthot * self.hashs  # (batch_size, input_size, input_level)
            x_hash = torch.sum(x_hash, dim=(1, 2))  # (batch_size,)

            x_indices = x_hash % (2**self.hash_logsize)  # (batch_size,)

        if self.sub_features > 1:
            x_features = []
            for i in range(self.sub_features):
                assert torch.all(
                    x_indices < self.offsets[i + 1] - self.offsets[i]
                ), f"indices overflow: {i}, {(x_indices.min(), x_indices.max())}, {(self.offsets[i], self.offsets[i+1])}"
                x_features.append(self.features(x_indices + self.offsets[i]))
                x_indices = torch.floor_divide(x_indices, self.sub_divisor)
            x_features = torch.cat(x_features, dim=1)  # (batch_size, level_dim * sub_features)
            x_features = self.feature_mapping(x_features)  # (batch_size, dim_feature)
        else:
            x_features = self.features(x_indices)  # (batch_size, dim_feature)

        if self.quant_int8:
            x_features = fake_quant(torch.clamp(x_features, min=-1, max=127 / 128), scale=128)
        return x_features


# ---------------------------------------------------------
# Switch Variant of Networks
    
class SwitchGate(nn.Module):
    """Switch Gating for MoE networks"""
    def __init__(self, num_experts: int, jitter_eps=0.0, no_scaling=False) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.jitter_eps = jitter_eps
        self.no_scaling = no_scaling

    def forward(self, route_logits: torch.Tensor) -> torch.Tensor:
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
        route_frac = torch.tensor([(route_idx == i).sum() * inv_batch_size
                                   for i in range(self.num_experts)],
                                  dtype=route_probs.dtype,
                                  device=route_probs.device)  # [num_experts]
        route_prob_mean = route_probs.mean(0)  # [num_experts]
        load_balancing_loss = self.num_experts * torch.dot(route_frac, route_prob_mean)
        load_balancing_loss = load_balancing_loss - 1.0

        if self.no_scaling:
            route_multiplier = route_prob_max / route_prob_max.detach()
        else:
            route_multiplier = route_prob_max

        aux_outputs = {
            'route_prob_max': route_prob_max,
            'route_frac_min': route_frac.min(),
            'route_frac_max': route_frac.max(),
            'route_frac_std': route_frac.std(),
        }
        return route_idx, route_multiplier, load_balancing_loss, aux_outputs
    

class SwitchLinear(nn.Module):
    '''Switchable linear layer for MoE networks'''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_experts: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert num_experts > 0, "Number of experts must be at least 1"
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.empty((num_experts, out_features * in_features), **factory_kwargs))
        self.weight_fact = nn.Parameter(
            torch.empty((1, out_features * in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts, out_features), **factory_kwargs))
            self.bias_fact = nn.Parameter(torch.empty((1, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
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

    def get_weight_and_bias(self, route_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = route_index.shape[0]
        expert_weight = F.embedding(route_index, self.weight) + self.weight_fact
        expert_weight = expert_weight.view(batch_size, self.out_features, self.in_features)
        if self.bias is not None:
            expert_bias = F.embedding(route_index, self.bias) + self.bias_fact
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def get_weight_and_bias_at_idx(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        expert_weight = self.weight[index] + self.weight_fact[0]
        expert_weight = expert_weight.view(self.out_features, self.in_features)
        if self.bias is not None:
            expert_bias = self.bias[index] + self.bias_fact[0]
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def forward(self, input: torch.Tensor, route_index: torch.Tensor) -> torch.Tensor:
        expert_weight, expert_bias = self.get_weight_and_bias(route_index)
        output = torch.einsum('bmn,bn->bm', expert_weight, input)
        if expert_bias is not None:
            output = output + expert_bias
        return output


class SwitchLinearBlock(nn.Module):
    '''LinearBlock with switchable linear layer for MoE networks'''
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_experts,
                 activation='relu',
                 bias=True,
                 quant=False,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_bits=32) -> None:
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

    def forward(self, x: torch.Tensor, route_index: torch.Tensor):
        if self.quant:
            weight, bias = self.fc.get_weight_and_bias(route_index)
            x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
            w = fake_quant(weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
            if bias is not None:
                b = fake_quant(bias,
                               self.weight_quant_scale * self.input_quant_scale,
                               num_bits=self.bias_quant_bits)
                out = F.linear(x, w, b)
            else:
                out = F.linear(x, w)
        else:
            out = self.fc(x, route_index)

        if self.activation:
            out = self.activation(out)
        return out


class SwitchConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_experts: int,
                 kernel_size: int,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
        factory_kwargs = {'device': device, 'dtype': dtype}
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
            self.register_parameter('bias', None)
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

    def get_weight_and_bias(self, route_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = route_index.shape[0]
        expert_weight = F.embedding(route_index, self.weight) + self.weight_fact
        expert_weight = expert_weight.view(batch_size, *self.weight_shape)
        if self.bias is not None:
            expert_bias = F.embedding(route_index, self.bias) + self.bias_fact
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def get_weight_and_bias_at_idx(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        expert_weight = (self.weight[index] + self.weight_fact[0]).view(self.weight_shape)
        if self.bias is not None:
            expert_bias = self.bias[index] + self.bias_fact[0]
        else:
            expert_bias = None
        return expert_weight, expert_bias

    def forward(self, input: torch.Tensor, route_index: torch.Tensor):
        if self.padding_mode != 'zeros':
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
        output = F.conv2d(input, expert_weight, expert_bias, self.stride, padding, self.dilation,
                          self.groups * batch_size)
        output = output.view(batch_size, self.weight_shape[0], output.size(-2), output.size(-1))
        return output


class SwitchConv2dBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_experts,
                 ks,
                 st=1,
                 padding=0,
                 activation='relu',
                 pad_type='zeros',
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
                 bias_quant_bits=32):
        super().__init__()
        assert pad_type in ['zeros', 'reflect', 'replicate',
                            'circular'], f"Unsupported padding mode: {pad_type}"
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
            assert self.conv.padding_mode == 'zeros', "quant conv requires zero padding"
            self.input_quant_scale = input_quant_scale
            self.input_quant_bits = input_quant_bits
            self.weight_quant_scale = weight_quant_scale
            self.weight_quant_bits = weight_quant_bits
            self.bias_quant_scale = bias_quant_scale or (input_quant_scale * weight_quant_scale)
            self.bias_quant_bits = bias_quant_bits

    def forward(self, x: torch.Tensor, route_index: torch.Tensor):
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
            x = F.conv2d(x, w, bias, self.conv.stride, self.conv.padding, self.conv.dilation,
                         self.conv.groups * batch_size)
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))
        else:
            x = self.conv(x, route_index)
        if not self.activation_first and self.activation:
            x = self.activation(x)
        return x


# ---------------------------------------------------------
# Custom Containers


class SequentialWithExtraArguments(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x
