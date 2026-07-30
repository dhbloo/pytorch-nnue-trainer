import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import PartialState
from accelerate.utils import gather, gather_object, reduce
from pykeops.torch import LazyTensor
from torch import Tensor

from .layers.linear import Linear
from .ops import (
    accelerated_cosine_argmin,
    accelerated_l2_argmin,
    accelerated_perplexity_stats,
)
from .ops.vector_quantization import _update_ema_codebook_


_DEAD_CODE_BOUND_EPS = 1e-6
_DEAD_CODE_SAFE_THRESHOLD_MULTIPLIER = 2
_DDP_ZERO_LINK_ONLY_ATTR = "_vq_ddp_zero_link_only"
_VQ_INIT_CONTINUATION_VERSION = 1
_VQ_FINITE_CHECK_CHUNK_ELEMENTS = 1024 * 1024


class _DDPPreinitIdentity(torch.autograd.Function):
    """Preserve a tensor exactly while making selected parameters graph-visible."""

    @staticmethod
    def forward(ctx, x, *parameters):
        ctx.save_for_backward(*parameters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, *(torch.zeros_like(parameter) for parameter in ctx.saved_tensors)


class _DDPRealUseIdentity(torch.autograd.Function):
    """Mark real use from a graph-local codebook identity."""

    @staticmethod
    def forward(ctx, tensor, parameters):
        ctx.parameters = parameters
        return tensor

    @staticmethod
    def backward(ctx, gradient):
        mark_ddp_parameter_use(ctx.parameters)
        return gradient, None


def _ddp_zero_link_enabled():
    return (
        torch.is_grad_enabled()
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def ddp_preinit_identity(x, parameters):
    """Attach a DDP-only zero-gradient link without changing ``x``."""
    if not _ddp_zero_link_enabled():
        return x
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameters:
        return x
    for parameter in parameters:
        if getattr(parameter, _DDP_ZERO_LINK_ONLY_ATTR, True):
            setattr(parameter, _DDP_ZERO_LINK_ONLY_ATTR, True)
    return _DDPPreinitIdentity.apply(x, *parameters)


def mark_ddp_parameter_use(parameters):
    """Record real use after an earlier zero-link in this accumulation window."""
    for parameter in parameters:
        if hasattr(parameter, _DDP_ZERO_LINK_ONLY_ATTR):
            setattr(parameter, _DDP_ZERO_LINK_ONLY_ATTR, False)


def mark_ddp_parameter_use_on_backward(tensor, parameters):
    """Record real parameter use only if backward reaches ``tensor``."""
    if not _ddp_zero_link_enabled() or not tensor.requires_grad:
        return tensor
    return _DDPRealUseIdentity.apply(tensor, tuple(parameters))


def clear_ddp_preinit_gradients(parameters):
    """Remove zero-link-only gradients before clipping or optimizer mutation."""
    for parameter in parameters:
        zero_link_only = getattr(parameter, _DDP_ZERO_LINK_ONLY_ATTR, None)
        if zero_link_only:
            parameter.grad = None
        if zero_link_only is not None:
            delattr(parameter, _DDP_ZERO_LINK_ONLY_ATTR)


def _vq_all_finite_chunked(tensor):
    flat = tensor.reshape(-1)
    for offset in range(0, flat.numel(), _VQ_FINITE_CHECK_CHUNK_ELEMENTS):
        if not torch.isfinite(
            flat[offset : offset + _VQ_FINITE_CHECK_CHUNK_ELEMENTS]
        ).all():
            return False
    return True


def l2_norm(x: Tensor) -> Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1e-7)


def squared_l2_dist(x: Tensor, y: Tensor) -> LazyTensor:
    """Compute pairwise squared L2 distances without materializing the matrix."""
    x_i = LazyTensor(x[:, None, :])  # (N, 1, dim_feature)
    y_j = LazyTensor(y[None, :, :])  # (1, M, dim_feature)
    return ((x_i - y_j) ** 2).sum(-1)  # (N, M)


def l2_dist(x: Tensor, y: Tensor) -> LazyTensor:
    """
    Compute the L2 distance between x and y.
    Args:
        x: Input embedding tensor of shape (N, dim_feature).
        y: Codebook tensor of shape (M, dim_feature).
    Returns:
        dists: A tensor of shape (N, M).
    """
    # compute distances of x and embeddings: (x - e)^2 = x^2 + e^2 - 2 e * x
    # dists = x.square().sum(-1, keepdim=True) + y.square().sum(-1) \
    #       - 2 * torch.matmul(x, y.t())  # (N, M)
    # dists = torch.clamp(dists, min=0).sqrt()  # (N, M)

    return squared_l2_dist(x, y).sqrt()


def cosine_dist(x: Tensor, y: Tensor) -> LazyTensor:
    """
    Compute the cosine similarity between x and y.
    Args:
        x: Input embedding tensor of shape (N, dim_feature), assumed to be unit vec.
        y: Codebook tensor of shape (M, dim_feature), assumed to be unit vec.
    Returns:
        dists: A tensor of shape (N, M).
    """
    x_i = LazyTensor(x[:, None, :])  # (N, 1, dim_feature)
    y_j = LazyTensor(y[None, :, :])  # (1, M, dim_feature)
    dists = 1 - (x_i * y_j).sum(-1)  # (N, M)
    return dists


def uniform_init(*shape) -> Tensor:
    """Initialize a tensor with uniform random values."""
    codebook = torch.empty(shape)
    nn.init.kaiming_uniform_(codebook)
    return codebook


def uniform_lazy_tensor_2d(dim0, dim1, device) -> LazyTensor:
    """Initialize a 2D lazy tensor with uniform random values."""
    rand_x = LazyTensor(torch.rand((dim0, 1, 1), device=device))
    rand_y = LazyTensor(torch.rand((1, dim1, 1), device=device))

    rand_xy = (rand_x * 12.9898 + rand_y * 78.233).sin() * 43758.5453123
    rand_xy_floor = (rand_xy - 0.5).round()
    rand_xy_fract = rand_xy - rand_xy_floor
    rand_xy_clamp = rand_xy_fract.clamp(0, 1)

    return rand_xy_clamp


def gumbel_sample(logits: LazyTensor, stochastic=False, temperature=1.0, training=True) -> Tensor:
    """
    Sample from a categorical distribution using the Gumbel-Softmax trick.
    Args:
        logits: Unnormalized log probabilities of shape (N, num_classes).
        stochastic: Whether to use Gumbel sampling.
        temperature: Temperature parameter for sampling.
        training: Whether the model is in training mode.
    Returns:
        ind: Indices of the sampled elements.
    """
    if training and stochastic and temperature > 0:
        device = logits.variables[0].device
        gumbel_noise = uniform_lazy_tensor_2d(*logits.shape, device=device)
        gumbel_noise = -(-(gumbel_noise * (1 - 2e-7) + 1e-7).log()).log()
        sampling_logits = (logits / temperature) + gumbel_noise
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=1).squeeze(1)  # (N,)
    return ind


def compute_perplexity(cluster_size: Tensor) -> Tensor:
    probs = cluster_size / cluster_size.sum()
    entropy = probs * torch.log(probs + 1e-10)
    perplexity = torch.exp(-torch.sum(entropy))
    return perplexity


def entropy_probability_sum(logits: LazyTensor, temperature=1.0) -> Tensor:
    logits = logits * (1.0 / temperature)  # (N, codebook_size)
    logits_max_and_exp = logits.reduction("Max_SumShiftExp", dim=1)  # (N, 2)
    logits_max = logits_max_and_exp[:, 0].contiguous()
    logits_sumexp = logits_max_and_exp[:, 1].contiguous()
    logits_exp = (logits - LazyTensor(logits_max[:, None], axis=0)).exp()  # (N, codebook_size)
    softprobs = logits_exp / LazyTensor(logits_sumexp[:, None], axis=0)  # (N, codebook_size)
    return softprobs.sum(0).squeeze(1)


def entropy_regularization_from_probability_sum(
    probability_sum: Tensor, count: int
) -> Tensor:
    avg_probs = probability_sum / count
    entropy = (-avg_probs * torch.log(avg_probs.clamp_min(1e-7))).sum(dim=-1)
    return -entropy


def entropy_regularization(logits: LazyTensor, temperature=1.0) -> Tensor:
    probability_sum = entropy_probability_sum(logits, temperature)
    return entropy_regularization_from_probability_sum(
        probability_sum, logits.shape[0]
    )


def sample_vectors(inputs: Tensor, num_samples: int) -> Tensor:
    """Sample num_samples vectors from the input tensor (supports DDP)."""
    num_processes = PartialState().num_processes
    process_idx = PartialState().process_index
    num_samples_per_rank = (num_samples + num_processes - 1) // num_processes

    num_inputs, device = inputs.shape[0], inputs.device
    assert num_inputs > 0, "Can not sample empty input tensor"
    if num_inputs >= num_samples_per_rank:
        indices = torch.randperm(num_inputs, device=device)[:num_samples_per_rank]
    else:
        indices = torch.randint(0, num_inputs, (num_samples_per_rank,), device=device)
    sampled = inputs[indices]  # (num_sample_per_rank, dim_feature)
    sampled = gather(sampled)  # (>=num_samples, dim_feature)
    assert sampled.shape[0] >= num_samples
    sampled = sampled[:num_samples]  # (num_samples, dim_feature)

    return sampled


def kmeans(
    inputs: Tensor,
    num_clusters: int,
    num_iters: int = 10,
    use_cosine_sim: bool = False,
) -> Tensor:
    """
    Perform K-means clustering on the input embeddings.
    This method supports working under distributed settings.
    Args:
        inputs: Input embeddings of shape (num_samples, dim_feature).
            If under distributed settings, the input tensor should have
            the same number of different samples across all processes, and
            all the inputs across processes will be used for clustering.
        num_clusters: The number of clusters.
        num_iters: The number of iterations.
        use_cosine_sim: Whether to use cosine similarity. Inputs should
            be normalized if this is set to True.
    Returns:
        means: The cluster centroids of shape (num_clusters, dim_feature).
        bins: The number of samples in each cluster of shape (num_clusters,).
    """
    (num_samples_per_rank, dim_feature), device, dtype = inputs.shape, inputs.device, inputs.dtype
    num_samples = num_samples_per_rank * PartialState().num_processes
    if num_samples < num_clusters:
        raise ValueError(f"Number of samples {num_samples} < number of clusters {num_clusters}")

    # initialize cluster centroids
    means = sample_vectors(inputs, num_clusters)  # (num_clusters, dim_feature)

    # perform K-means iterations
    for _ in range(num_iters):
        if use_cosine_sim:
            dist = cosine_dist(inputs, means)  # (num_samples_per_rank, num_clusters)
        else:
            dist = l2_dist(inputs, means)  # (num_samples_per_rank, num_clusters)

        buckets = dist.argmin(dim=1).squeeze(1)  # (num_samples_per_rank,)
        bins = torch.bincount(buckets, minlength=num_clusters)  # (num_clusters,)
        bins = reduce(bins, reduction="sum")  # (num_clusters,)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = torch.zeros(num_clusters, dim_feature, device=device, dtype=dtype)
        new_means.scatter_add_(0, buckets[:, None].expand(-1, dim_feature), inputs)
        new_means = reduce(new_means, reduction="sum")  # (num_clusters, dim_feature)
        new_means = new_means / bins_min_clamped[:, None]  # (num_clusters, dim_feature)

        if use_cosine_sim:
            new_means = l2_norm(new_means)

        means = torch.where(zero_mask[:, None], means, new_means)

    return means, bins


def ema_inplace(tensor: Tensor, new: Tensor, decay: float) -> Tensor:
    return tensor.lerp_(new, 1 - decay)


def efficient_rotation_trick_transform(u: Tensor, q: Tensor, e: Tensor) -> Tensor:
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    w = l2_norm(u + q).detach()  # (N, dim_feature)
    # These are vector projections. Expressing them as reductions avoids
    # launching two batches of tiny 1xD matrix multiplications.
    return (
        e
        - 2 * (e * w).sum(dim=-1, keepdim=True) * w
        + 2 * (e * u.detach()).sum(dim=-1, keepdim=True) * q.detach()
    )


def rotate_to(src: Tensor, tgt: Tensor):
    """
    Rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    Inputs:
        src: Source tensor of shape (N, dim_feature).
        tgt: Target tensor of shape (N, dim_feature).
    """
    norm_src = src.norm(dim=-1, keepdim=True)
    norm_tgt = tgt.norm(dim=-1, keepdim=True)

    rotated_tgt = efficient_rotation_trick_transform(
        src / norm_src.clamp(min=1e-6),
        tgt / norm_tgt.clamp(min=1e-6),
        src,
    )

    rotated = rotated_tgt * (norm_tgt / norm_src.clamp(min=1e-6)).detach()

    return rotated


def list_mean(xs: list):
    if len(xs) == 0:
        raise ValueError("Cannot compute mean of empty list")
    if len(xs) == 1:
        return xs[0]
    sum = xs[0] + xs[1]
    for i in range(2, len(xs)):
        sum += xs[i]
    return sum / len(xs)


class VectorQuantize(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Args:
        codebook_size: The number of embeddings in the codebook.
        dim_feature: Dimension of the embeddings.
        beta: commitment cost used in loss term, beta * ||x_q-sg(x)||^2
        accelerated_search: Use a guarded Tensor-Core shortlist when available.
    """

    def __init__(
        self,
        codebook_size: int,
        dim_feature: int,
        use_cosine_sim=False,
        kmeans_init=True,
        kmeans_sample_multiplier=1,
        kmeans_iter=10,
        ema_update=True,
        ema_decay=0.995,
        threshold_ema_dead_code=5e-3,
        reset_cluster_size=None,
        commitment_weight=0.25,
        learnable_codebook=True,
        rotation_trick=False,
        stochastic_sampling=False,
        sampling_temp=1.0,
        entropy_reg_weight=0.0,
        entropy_reg_temp=0.01,
        use_simvq=False,
        codebook_transform=None,
        convert_to_fp32=True,
        accelerated_search=True,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim_feature = dim_feature
        self.use_cosine_sim = use_cosine_sim
        self.kmeans_init = kmeans_init
        self.kmeans_sample_multiplier = kmeans_sample_multiplier
        self.kmeans_iter = kmeans_iter
        self.ema_update = ema_update and not use_simvq
        self.ema_decay = ema_decay
        self.threshold_ema_dead_code = 0 if use_simvq else threshold_ema_dead_code
        self.reset_cluster_size = 1 if reset_cluster_size is None else reset_cluster_size
        self.commitment_weight = commitment_weight
        self.rotation_trick = rotation_trick
        self.stochastic_sampling = stochastic_sampling
        self.sampling_temp = sampling_temp
        self.entropy_reg_weight = entropy_reg_weight
        self.entropy_reg_temp = entropy_reg_temp
        self.use_simvq = use_simvq
        self.convert_to_fp32 = convert_to_fp32
        self.accelerated_search = accelerated_search
        self.accelerated_search_fallback_count = 0
        # Conservative single-rank host bounds let the normal FP32 EMA path
        # (0 <= decay <= 1) prove that no code can expire without synchronizing
        # a CUDA boolean every step.
        # They are derived from persistent buffers and never enter state_dict;
        # DDP retains the exact distributed check because unique counts can
        # differ across ranks.
        self._invalidate_dead_code_cache()

        self.register_buffer("inited", torch.zeros([], dtype=torch.bool))
        self._inited_python = None
        self._init_consensus_checked = False
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self._eval_cluster_size = None
        self._collect_eval_stats = True
        self._eval_entry_mask = None
        if kmeans_init:
            embed = torch.zeros(codebook_size, dim_feature)
            num_processes = PartialState().num_processes
            self._init_input_samples_target_per_process = (
                codebook_size * kmeans_sample_multiplier + num_processes - 1
            ) // num_processes
            self._init_input_samples_per_process = (
                self._init_input_samples_target_per_process
            )
            self._init_input_batches_this_process = []
            self._init_input_count_this_process = 0
        elif use_simvq:
            embed = torch.randn(codebook_size, dim_feature) * (dim_feature**-0.5)
            self.inited.data.fill_(True)
        else:
            embed = uniform_init(codebook_size, dim_feature)
            if use_cosine_sim:
                embed = l2_norm(embed)
            self.inited.data.fill_(True)

        if use_simvq:
            if codebook_transform is None:
                codebook_transform = Linear(dim_feature, dim_feature, bias=False)
                if kmeans_init:
                    nn.init.eye_(codebook_transform.weight)
            self.code_transform = codebook_transform

        self.learnable_codebook = learnable_codebook and not self.ema_update
        self.embed = nn.Parameter(embed, requires_grad=self.learnable_codebook)
        self.register_buffer("embed_avg", embed.clone())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Persistent buffers may have changed during a parent model load.
        # Refresh their host-side hot-path caches on the next forward.
        self._inited_python = None
        self._init_consensus_checked = False
        self._invalidate_dead_code_cache()

    def _invalidate_dead_code_cache(self):
        self._dead_code_min_cluster_lower_python = None
        self._dead_code_total_upper_python = None

    def is_initialized(self) -> bool:
        """Return the initialization state through its lazy host-side cache."""
        if self._inited_python is None:
            self._inited_python = bool(self.inited.item())
        return self._inited_python

    def _vq_init_continuation_descriptor(self):
        return {
            "codebook_size": self.codebook_size,
            "dim_feature": self.dim_feature,
            "kmeans_sample_multiplier": self.kmeans_sample_multiplier,
            "kmeans_iter": self.kmeans_iter,
            "use_cosine_sim": self.use_cosine_sim,
            "use_simvq": self.use_simvq,
            "convert_to_fp32": self.convert_to_fp32,
            "parameter_dtype": str(self.embed.dtype),
            "target_samples": getattr(
                self,
                "_init_input_samples_target_per_process",
                0,
            ),
        }

    def _vq_init_continuation_state(self):
        """Export rank-local pending k-means inputs outside ``state_dict``."""
        initialized = torch.equal(
            self.inited.detach(),
            self.inited.new_ones(()),
        )
        batches = (
            []
            if initialized
            else self._init_input_batches_this_process
        )
        count = (
            0
            if initialized
            else self._init_input_count_this_process
        )
        target = self._vq_init_continuation_descriptor()["target_samples"]
        if not 0 <= count <= target:
            raise ValueError(
                f"pending VQ sample count {count} is outside [0, {target}]"
            )
        if sum(len(batch) for batch in batches) != count:
            raise ValueError("pending VQ sample batches do not match their count")
        dtype = (
            batches[0].dtype
            if batches
            else (
                torch.float32
                if self.convert_to_fp32
                else self.embed.dtype
            )
        )
        samples = torch.empty(
            (count, self.dim_feature),
            dtype=dtype,
            device="cpu",
        )
        offset = 0
        for batch in batches:
            if (
                batch.ndim != 2
                or batch.shape[1] != self.dim_feature
                or batch.dtype != dtype
            ):
                raise ValueError(
                    "pending VQ sample batch shape or dtype is inconsistent"
                )
            samples[offset : offset + len(batch)].copy_(
                batch.detach(),
                non_blocking=False,
            )
            offset += len(batch)
        if not _vq_all_finite_chunked(samples):
            raise ValueError("pending VQ samples contain non-finite values")
        return {
            "version": _VQ_INIT_CONTINUATION_VERSION,
            "descriptor": self._vq_init_continuation_descriptor(),
            "dtype": str(dtype),
            "inited": initialized,
            "count": count,
            "samples": samples,
        }

    def _load_vq_init_continuation_state(self, state):
        """Restore rank-local pending k-means inputs after device placement."""
        if not isinstance(state, dict):
            raise ValueError("VQ continuation payload must be a dictionary")
        if set(state) != {
            "version",
            "descriptor",
            "dtype",
            "inited",
            "count",
            "samples",
        }:
            raise ValueError(
                "VQ continuation payload has missing or unexpected fields"
            )
        if state.get("version") != _VQ_INIT_CONTINUATION_VERSION:
            raise ValueError(
                f"unsupported VQ continuation version {state.get('version')!r}"
            )
        descriptor = state.get("descriptor")
        expected_descriptor = self._vq_init_continuation_descriptor()
        if descriptor != expected_descriptor:
            raise ValueError(
                "VQ continuation descriptor differs: "
                f"saved={descriptor!r}, current={expected_descriptor!r}"
            )
        initialized = state.get("inited")
        if type(initialized) is not bool:
            raise ValueError("VQ continuation inited flag must be a bool")
        loaded_initialized = torch.equal(
            self.inited.detach(),
            self.inited.new_full((), initialized),
        )
        if not loaded_initialized:
            raise ValueError(
                "VQ continuation inited flag differs from loaded model state"
            )
        count = state.get("count")
        target = expected_descriptor["target_samples"]
        if type(count) is not int or not 0 <= count <= target:
            raise ValueError(
                f"VQ continuation count {count!r} is outside [0, {target}]"
            )
        samples = state.get("samples")
        if not isinstance(samples, Tensor):
            raise ValueError("VQ continuation samples must be a tensor")
        if samples.device.type != "cpu" or not samples.is_contiguous():
            raise ValueError(
                "VQ continuation samples must be contiguous on CPU"
            )
        if tuple(samples.shape) != (count, self.dim_feature):
            raise ValueError(
                "VQ continuation sample shape differs from count/feature dimension"
            )
        if str(samples.dtype) != state.get("dtype"):
            raise ValueError("VQ continuation sample dtype metadata differs")
        if self.convert_to_fp32 and samples.dtype != torch.float32:
            raise ValueError("FP32-converting VQ requires FP32 pending samples")
        if not samples.is_floating_point():
            raise ValueError("VQ continuation samples must be floating point")
        if not _vq_all_finite_chunked(samples):
            raise ValueError("VQ continuation samples contain non-finite values")
        if initialized and count:
            raise ValueError("initialized VQ cannot restore pending samples")

        self._inited_python = None
        if initialized:
            return
        restored = samples.to(device=self.embed.device)
        self._init_input_batches_this_process = [restored] if count else []
        self._init_input_count_this_process = count
        self._init_consensus_checked = False

    def reset_eval_perplexity_stats(self):
        """Reset code-usage counts accumulated by subsequent evaluation forwards."""
        self._eval_cluster_size = self.cluster_size.new_zeros(
            self.cluster_size.shape,
            dtype=torch.long,
        )
        self._collect_eval_stats = True

    def set_eval_perplexity_stats_enabled(self, enabled: bool):
        """Include or exclude subsequent evaluation forwards from code-usage counts."""
        self._collect_eval_stats = enabled

    def set_eval_entry_mask(self, mask):
        """Select quantizer input entries that represent real evaluation rows."""
        self._eval_entry_mask = (
            None if mask is None else mask.to(device=self.embed.device, dtype=torch.bool)
        )

    def eval_perplexity_from_cluster_size(self, cluster_size: Tensor):
        """Compute this codebook's perplexity metrics from global usage counts."""
        perplexity = compute_perplexity(cluster_size)
        return perplexity, perplexity / self.codebook_size

    def _accumulate_input_batch(self, x):
        num_samples_remain = self._init_input_samples_per_process - self._init_input_count_this_process
        if num_samples_remain > 0:
            if x.shape[0] > num_samples_remain:
                x = x[:num_samples_remain]
            x = x.detach().clone()
            self._init_input_batches_this_process.append(x)
            self._init_input_count_this_process += x.shape[0]

        # We need to make sure all process have enough inputs before we continue
        has_enough_inputs = self._init_input_count_this_process >= self._init_input_samples_per_process
        has_enough_inputs = gather_object([has_enough_inputs])
        if not all(has_enough_inputs):
            return  # wait for more inputs

        inputs = torch.cat(self._init_input_batches_this_process, dim=0)  # (num_samples, dim_feature)
        # free up memory
        del self._init_input_samples_per_process
        del self._init_input_batches_this_process
        del self._init_input_count_this_process
        self._init_embed(inputs)

    def _check_first_forward_consensus(self, x):
        if (
            self._init_consensus_checked
            or PartialState().num_processes <= 1
        ):
            return
        descriptor = {
            "kmeans_init": self.kmeans_init,
            "initialized": self._inited_python,
            "codebook_size": self.codebook_size,
            "dim_feature": self.dim_feature,
            "kmeans_sample_multiplier": self.kmeans_sample_multiplier,
            "kmeans_iter": self.kmeans_iter,
            "target_samples": getattr(
                self,
                "_init_input_samples_target_per_process",
                0,
            ),
            "use_cosine_sim": self.use_cosine_sim,
            "use_simvq": self.use_simvq,
            "ema_update": self.ema_update,
            "threshold_ema_dead_code": self.threshold_ema_dead_code,
            "reset_cluster_size": self.reset_cluster_size,
            "training": self.training,
            "input_dtype": str(x.dtype),
            "input_device": x.device.type,
        }
        descriptors = gather_object([descriptor])
        if any(value != descriptors[0] for value in descriptors[1:]):
            raise RuntimeError(
                "VQ first-forward configuration or initialization status "
                "differs across ranks: "
                + "; ".join(
                    f"rank {rank}={value!r}"
                    for rank, value in enumerate(descriptors)
                )
            )
        self._init_consensus_checked = True

    @torch.no_grad()
    def _init_embed(self, inputs):
        num_total_inputs = inputs.shape[0] * PartialState().num_processes
        assert (
            num_total_inputs >= self.codebook_size
        ), f"Number of inputs {num_total_inputs} < codebook size {self.codebook_size}"

        if num_total_inputs > self.codebook_size:
            embed, cluster_size = kmeans(
                inputs=inputs,
                num_clusters=self.codebook_size,
                num_iters=self.kmeans_iter,
                use_cosine_sim=self.use_cosine_sim,
            )

            cluster_size = cluster_size.float()
            embed_sum = embed * cluster_size[:, None]
        else:
            embed = gather(inputs)
            cluster_size = torch.ones(self.codebook_size, device=inputs.device)
            embed_sum = embed

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.fill_(True)
        self._inited_python = True
        self._invalidate_dead_code_cache()

    @torch.no_grad()
    def _expire_codes(self, inputs) -> Tensor | None:
        if self.threshold_ema_dead_code == 0:
            return None

        if PartialState().num_processes == 1:
            min_cluster = self._dead_code_min_cluster_lower_python
            total_upper = self._dead_code_total_upper_python
            if min_cluster is not None and total_upper is not None:
                normalized_lower_bound = min_cluster * self.codebook_size / total_upper
                # A wide margin absorbs FP32 lerp/reduction rounding while the
                # cached bounds below are advanced conservatively.
                if normalized_lower_bound >= (
                    _DEAD_CODE_SAFE_THRESHOLD_MULTIPLIER
                    * self.threshold_ema_dead_code
                ):
                    return None

            stats = torch.stack((self.cluster_size.min(), self.cluster_size.sum()))
            min_cluster, cluster_total = stats.tolist()
            normalized_min = min_cluster * self.codebook_size / cluster_total
            if normalized_min >= (
                _DEAD_CODE_SAFE_THRESHOLD_MULTIPLIER * self.threshold_ema_dead_code
            ):
                self._dead_code_min_cluster_lower_python = min_cluster * (
                    1 - _DEAD_CODE_BOUND_EPS
                )
                self._dead_code_total_upper_python = cluster_total * (
                    1 + _DEAD_CODE_BOUND_EPS
                )
                return None

        expired_codes = self.normalized_cluster_size < self.threshold_ema_dead_code
        if not expired_codes.any():
            return None

        num_expired_codes = expired_codes.sum().float()
        num_samples = int(num_expired_codes.item())
        sampled = sample_vectors(inputs, num_samples)  # (num_samples, dim_feature)
        reset_cluster_size = torch.full(
            (num_samples,), self.reset_cluster_size, dtype=torch.float, device=sampled.device
        )

        self.embed.data[expired_codes] = sampled
        self.embed_avg.data[expired_codes] = sampled * reset_cluster_size[:, None]
        self.cluster_size.data[expired_codes] = reset_cluster_size
        self._invalidate_dead_code_cache()

        return num_expired_codes

    @torch.no_grad()
    def _update_ema(self, x, embed_indices, cluster_size):
        cluster_size = cluster_size.float()
        ema_inplace(self.cluster_size, cluster_size, self.ema_decay)
        if (
            PartialState().num_processes == 1
            and self._dead_code_min_cluster_lower_python is not None
        ):
            decay = self.ema_decay
            self._dead_code_min_cluster_lower_python *= decay * (
                1 - _DEAD_CODE_BOUND_EPS
            )
            self._dead_code_total_upper_python = (
                self._dead_code_total_upper_python * decay
                + x.shape[0] * (1 - decay)
            ) * (1 + _DEAD_CODE_BOUND_EPS)

        if self.ema_update:
            if PartialState().num_processes == 1:
                # On one rank, update the persistent accumulator directly.
                # This avoids zeroing and then rereading a codebook-sized
                # embed_sum buffer. Scaling each contribution before the
                # atomic reduction changes only FP32 accumulation order.
                self.embed_avg.mul_(self.ema_decay)
                self.embed_avg.index_add_(
                    0,
                    embed_indices,
                    x,
                    alpha=1 - self.ema_decay,
                )
            else:
                embed_sum = torch.zeros_like(self.embed_avg)  # (codebook_size, dim_feature)
                embed_sum.scatter_add_(0, embed_indices[:, None].expand(-1, self.dim_feature), x)

                embed_sum = reduce(embed_sum, reduction="sum")  # (codebook_size, dim_feature)
                ema_inplace(self.embed_avg, embed_sum, self.ema_decay)

            # The fused normalization improves complete cosine-VQ steps. Keep
            # L2 on the native path, which is faster in the complete Mix graph.
            if not self.use_cosine_sim or not _update_ema_codebook_(
                self.embed,
                self.embed_avg,
                self.cluster_size,
                self.use_cosine_sim,
            ):

                def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
                    x_sum = x.sum(dim=dim, keepdim=True)
                    return x_sum * (x + eps) / (x_sum + n_categories * eps)

                cluster_size_smoothed = laplace_smoothing(self.cluster_size, self.codebook_size)
                embed_normalized = self.embed_avg / cluster_size_smoothed[:, None]
                if self.use_cosine_sim:
                    embed_normalized = l2_norm(embed_normalized)
                self.embed.data.copy_(embed_normalized)

    def _loss(
        self, x, quantized, dists, entropy_probability_sum_value=None
    ):
        # compute VQ-VAE loss
        loss_embed = (quantized - x.detach()).square().mean()
        loss_commitment = (quantized.detach() - x).square().mean()
        loss = loss_embed + self.commitment_weight * loss_commitment
        loss_terms = {
            "loss_embed": loss_embed,
            "loss_commitment": loss_commitment,
        }

        # add entropy regularization loss
        if self.entropy_reg_weight > 0:
            entropy_reg_loss = (
                entropy_regularization(
                    -dists, self.entropy_reg_temp
                )
                if entropy_probability_sum_value is None
                else entropy_regularization_from_probability_sum(
                    entropy_probability_sum_value, len(x)
                )
            )
            loss = loss + self.entropy_reg_weight * entropy_reg_loss
            loss_terms["loss_entropy"] = entropy_reg_loss

        return loss, loss_terms

    @property
    def codebook(self) -> Tensor:
        """
        Returns:
            codebook: features of shape (codebook_size, dim_feature).
        """
        if self.use_simvq:
            code = self.code_transform(self.embed)
        else:
            code = self.embed

        if self.use_cosine_sim and not self.ema_update:
            code = l2_norm(code)

        return code

    @property
    def normalized_cluster_size(self) -> Tensor:
        """
        Returns:
            normalized_cluster_size: Normalized cluster size of shape (codebook_size,).
        """
        return self.cluster_size * (self.codebook_size / self.cluster_size.sum())

    def from_indices(self, indices: Tensor) -> Tensor:
        """
        Args:
            indices: Indices of the embedding, long tensor of shape (N,).
        Returns:
            x_q: A quantized tensor of shape (N, dim_feature).
        """
        return F.embedding(indices, self.codebook)  # (..., dim_feature)

    @torch.compiler.disable
    def forward(self, x: Tensor, input_normalized: bool = False):
        """
        Get the quantized version x_q of the continuous input x.
        Args:
            x: A continuous tensor of shape (N, dim_feature).
            input_normalized: Whether the input is already normalized to unit sphere.
        Returns:
            x_q: A quantized tensor of shape (N, dim_feature).
            info: A dictionary containing the following:
                dists: The distances to the codebook embeddings, float tensor of shape (N, codebook_size).
                embed_indices: Indices of the closest embedding, long tensor of shape (N,).
                embed_onehot: One-hot tensor of the closest embedding, float tensor of shape (N, codebook_size).
                perplexity: The perplexity of the embeddings, scalar.
                normalized_perplexity: The normalized perplexity in [0, 1], scalar.
                loss: The total VQ-VAE loss including embed and commitment terms, scalar.
        """
        assert x.ndim == 2, f"Wrong input ndim {x.ndim} != 2"
        assert x.shape[-1] == self.dim_feature, f"Wrong input dim {x.shape[-1]} != {self.dim_feature}"

        if self.convert_to_fp32:
            x = x.float()

        if self.use_cosine_sim and not input_normalized:
            x = l2_norm(x)

        initialized = self.is_initialized()
        self._check_first_forward_consensus(x)
        if not initialized:
            if self.training:
                self._accumulate_input_batch(x)
            return ddp_preinit_identity(x, self.parameters()), None

        # expire codes if we are in training mode
        num_expired_codes = None
        if self.training:
            num_expired_codes = self._expire_codes(x)
        if num_expired_codes is None:
            num_expired_codes = torch.zeros((), device=x.device)

        # get codebook embeddings
        codebook = mark_ddp_parameter_use_on_backward(
            self.codebook,
            self.parameters(),
        )  # (codebook_size, dim_feature)

        # compute distances to codebook embeddings
        stochastic_search = self.training and self.stochastic_sampling and self.sampling_temp > 0
        if self.use_cosine_sim:
            dists = cosine_dist(x, codebook)  # (N, codebook_size)
            search_dists = dists
        else:
            squared_dists = squared_l2_dist(x, codebook)
            dists = squared_dists.sqrt()  # Preserve the public distance semantics.
            search_dists = dists if stochastic_search else squared_dists

        # find closest codebook embeddings
        embed_indices = None
        accelerated_search_requested = (
            not stochastic_search and self.accelerated_search
        )
        if accelerated_search_requested:
            if self.use_cosine_sim:
                embed_indices = accelerated_cosine_argmin(x, codebook)
            else:
                embed_indices = accelerated_l2_argmin(x, codebook)
        if embed_indices is None:
            if accelerated_search_requested:
                self.accelerated_search_fallback_count += 1
            embed_indices = gumbel_sample(
                logits=-search_dists,
                stochastic=self.stochastic_sampling,
                temperature=self.sampling_temp,
                training=self.training,
            )  # (N,)

        # get quantized vectors
        quantized = F.embedding(embed_indices, codebook)  # (N, dim_feature)

        # preserve gradients
        if self.rotation_trick:
            x_q = rotate_to(x, quantized)
        else:  # standard STE to get gradients through VQ layer.
            x_q = x + (quantized - x).detach()

        eval_mask = None
        if not self.training and self._eval_entry_mask is not None:
            eval_mask = self._eval_entry_mask
            if eval_mask.ndim != 1 or len(eval_mask) != len(embed_indices):
                raise ValueError(
                    "evaluation VQ mask must contain one flag per quantizer entry"
                )

        # Compute evaluation statistics and losses from real entries only.
        stats_indices = (
            embed_indices[eval_mask] if eval_mask is not None else embed_indices
        )
        cluster_size = torch.bincount(
            stats_indices.view(-1), minlength=self.codebook_size
        )
        if self.training:
            cluster_size = reduce(cluster_size, reduction="sum")
        elif self._collect_eval_stats:
            if self._eval_cluster_size is None:
                self.reset_eval_perplexity_stats()
            self._eval_cluster_size.add_(cluster_size)
        stats_count = len(stats_indices)
        perplexity_stats = (
            accelerated_perplexity_stats(cluster_size, stats_count)
            if stats_count > 0 and PartialState().num_processes == 1
            else None
        )
        if stats_count == 0:
            perplexity = cluster_size.new_zeros((), dtype=x.dtype)
            normalized_perplexity = perplexity
        elif perplexity_stats is None:
            perplexity = compute_perplexity(cluster_size)
            normalized_perplexity = perplexity / self.codebook_size
        else:
            perplexity, normalized_perplexity = perplexity_stats

        # compute VQ losses
        selected_loss_dists = None
        selected_probability_sum = None
        if eval_mask is not None:
            if stats_count:
                loss_x = x[eval_mask]
                loss_quantized = quantized[eval_mask]
                if self.entropy_reg_weight > 0:
                    selected_loss_dists = (
                        cosine_dist(loss_x, codebook)
                        if self.use_cosine_sim
                        else squared_l2_dist(loss_x, codebook).sqrt()
                    )
                    selected_probability_sum = entropy_probability_sum(
                        -selected_loss_dists, self.entropy_reg_temp
                    )
                total_loss, loss_terms = self._loss(
                    loss_x,
                    loss_quantized,
                    selected_loss_dists,
                    selected_probability_sum,
                )
            else:
                total_loss = x.sum() * 0
                loss_terms = {
                    "loss_embed": total_loss,
                    "loss_commitment": total_loss,
                }
                if self.entropy_reg_weight > 0:
                    loss_terms["loss_entropy"] = total_loss
        else:
            if not self.training and self.entropy_reg_weight > 0:
                selected_loss_dists = dists
                selected_probability_sum = entropy_probability_sum(
                    -selected_loss_dists, self.entropy_reg_temp
                )
            total_loss, loss_terms = self._loss(
                x, quantized, dists, selected_probability_sum
            )

        loss_stats = None
        if not self.training:
            stats_x = x if eval_mask is None else x[eval_mask]
            stats_quantized = quantized if eval_mask is None else quantized[eval_mask]
            embed_sum = (stats_quantized - stats_x.detach()).square().sum()
            commit_sum = (stats_quantized.detach() - stats_x).square().sum()
            loss_stats = {
                "embed_sum": embed_sum,
                "commit_sum": commit_sum,
                "elements": int(stats_x.numel()),
                "vectors": int(len(stats_x)),
                "commitment_weight": float(self.commitment_weight),
                "entropy_weight": float(self.entropy_reg_weight),
            }
            if self.entropy_reg_weight > 0:
                if len(stats_x):
                    probability_sum = selected_probability_sum
                else:
                    probability_sum = x.new_zeros(self.codebook_size)
                loss_stats["probability_sum"] = probability_sum

        # gather inference and loss statistics
        info = {
            "dists": dists,
            "embed_indices": embed_indices,
            "perplexity": perplexity,
            "normalized_perplexity": normalized_perplexity,
            "num_expired_codes": num_expired_codes,
            "loss": total_loss,
            "loss_stats": loss_stats,
            **loss_terms,
        }

        # perform training EMA update and loss computation
        if self.training:
            self._update_ema(x, embed_indices, cluster_size)

        return x_q, info


class ProductVectorQuantize(nn.Module):
    """
    Product Vector Quantization (PVQ) layer.
    Args:
        codebook_size: The number of embeddings in each codebook.
        dim_feature: Total dimension of the embeddings.
        num_codebooks: Number of groups to split the input into.
    """

    def __init__(self, codebook_size, dim_feature, num_codebooks, **kwargs):
        super().__init__()
        assert dim_feature % num_codebooks == 0, "dim_feature must be divisible by num_codebooks"
        self.codebook_size = codebook_size
        self.dim_feature = dim_feature
        self.num_codebooks = num_codebooks
        self.dim_feature_per_group = dim_feature // num_codebooks
        self.vq_modules = nn.ModuleList(
            [
                VectorQuantize(
                    codebook_size=codebook_size,
                    dim_feature=self.dim_feature_per_group,
                    **kwargs,
                )
                for _ in range(num_codebooks)
            ]
        )

    @property
    def codebook(self) -> Tensor:
        """
        Returns:
            codebook: features of shape (num_codebooks, codebook_size, dim_feature_per_group).
        """
        return torch.stack([vq.codebook for vq in self.vq_modules], dim=0)

    @property
    def normalized_cluster_size(self) -> Tensor:
        """
        Returns:
            normalized_cluster_size: Normalized cluster size of shape (num_codebooks, codebook_size,).
        """
        return torch.stack([vq.normalized_cluster_size for vq in self.vq_modules], dim=0)

    def from_indices(self, indices: Tensor) -> Tensor:
        """
        Args:
            indices: Indices of the embedding, long tensor of shape (N, num_codebooks).
        Returns:
            x_q: A quantized tensor of shape (N, dim_feature).
        """
        assert indices.ndim == 2, f"Wrong input ndim {indices.ndim} != 2"
        assert (
            indices.shape[1] == self.num_codebooks
        ), f"Wrong input dim {indices.shape[-1]} != {self.num_codebooks}"
        x_q_groups = []
        for i, vq in enumerate(self.vq_modules):
            x_q = vq.from_indices(indices[:, i])
            x_q_groups.append(x_q)
        x_q = torch.cat(x_q_groups, dim=-1)
        return x_q

    def forward(self, x: Tensor):
        assert x.ndim == 2, f"Wrong input ndim {x.ndim} != 2"
        assert x.shape[1] == self.dim_feature, f"Wrong input dim {x.shape[-1]} != {self.dim_feature}"
        x_groups = x.view(-1, self.num_codebooks, self.dim_feature_per_group)
        x_groups = x_groups.unbind(1)  # num_codebooks * (N, dim_feature_per_group)

        x_q_groups = []
        info_groups = []
        for i, vq in enumerate(self.vq_modules):
            x_q, info = vq(x_groups[i].contiguous())
            x_q_groups.append(x_q)
            info_groups.append(info)

        x_q = torch.stack(x_q_groups, dim=1)  # (N, num_codebooks, dim_feature_per_group)
        x_q = x_q.view(-1, self.dim_feature)  # (N, dim_feature)

        aggregated_info, raw_info = None, None
        if all(info is not None for info in info_groups):
            raw_info = {k: [info_groups[i][k] for i in range(self.num_codebooks)] for k in info_groups[0]}

            mean_attributes = ["perplexity", "normalized_perplexity"]
            mean_attributes += [
                key
                for key in raw_info.keys()
                if key.startswith("loss") and key != "loss_stats"
            ]

            aggregated_info = {key: list_mean(raw_info[key]) for key in mean_attributes}
            aggregated_info["embed_indices"] = torch.stack(
                raw_info["embed_indices"], dim=1
            )  # (N, num_codebooks)
            aggregated_info["num_expired_codes"] = torch.sum(
                torch.stack([info["num_expired_codes"] for info in info_groups])
            )
            aggregated_info["loss_stats"] = raw_info["loss_stats"]

        return x_q, aggregated_info, raw_info
