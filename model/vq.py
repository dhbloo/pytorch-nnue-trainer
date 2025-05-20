import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import PartialState
from accelerate.utils import gather, reduce
from pykeops.torch import LazyTensor
from torch import Tensor


def l2_norm(x: Tensor) -> Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1e-7)


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

    x_i = LazyTensor(x[:, None, :])  # (N, 1, dim_feature)
    y_j = LazyTensor(y[None, :, :])  # (1, M, dim_feature)
    dists = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (N, M)
    return dists


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


def entropy_regularization(logits: LazyTensor, temperature=1.0) -> Tensor:
    logits = logits * (1.0 / temperature)  # (N, codebook_size)
    logits_max_and_exp = logits.reduction("Max_SumShiftExp", dim=1)  # (N, 2)
    logits_max = logits_max_and_exp[:, 0].contiguous()
    logits_sumexp = logits_max_and_exp[:, 1].contiguous()
    logits_exp = (logits - LazyTensor(logits_max[:, None], axis=0)).exp()  # (N, codebook_size)
    softprobs = logits_exp / LazyTensor(logits_sumexp[:, None], axis=0)  # (N, codebook_size)
    avg_probs = (softprobs.sum(0) / softprobs.shape[0]).squeeze(1)  # (codebook_size,)
    entropy = (-avg_probs * torch.log(avg_probs.clamp_min(1e-7))).sum(dim=-1)
    return -entropy


def sample_vectors(inputs: Tensor, num_samples: int) -> Tensor:
    """Sample num_samples vectors from the input tensor (supports DDP)."""
    num_processes = PartialState().num_processes
    process_idx = PartialState().process_index
    num_samples_per_rank, num_samples_remain = divmod(num_samples, num_processes)
    if process_idx < num_samples_remain:
        num_samples_per_rank += 1

    num_inputs, device = inputs.shape[0], inputs.device
    assert num_inputs > 0, "Can not sample empty input tensor"
    if num_inputs >= num_samples_per_rank:
        indices = torch.randperm(num_inputs, device=device)[:num_samples_per_rank]
    else:
        indices = torch.randint(0, num_inputs, (num_samples_per_rank,), device=device)
    sampled = inputs[indices]  # (num_sample_per_rank, dim_feature)
    sampled = gather(sampled)  # (num_samples, dim_feature)
    assert sampled.shape[0] == num_samples

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
    e = e[:, None, :]  # (N, 1, dim_feature)
    w = l2_norm(u + q).detach()  # (N, dim_feature)
    return (
        e
        - 2 * (e @ w[:, :, None] @ w[:, None, :])
        + 2 * (e @ u[:, :, None].detach() @ q[:, None, :].detach())
    ).squeeze(1)


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

        self.register_buffer("inited", torch.zeros([], dtype=torch.bool))
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        if kmeans_init:
            embed = torch.zeros(codebook_size, dim_feature)
            self._init_input_samples = (
                codebook_size * kmeans_sample_multiplier // PartialState().num_processes
            )
            self._init_input_batches = []
            self._init_input_count = 0
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
                codebook_transform = nn.Linear(dim_feature, dim_feature, bias=False)
                if kmeans_init:

                    def initialize(self):
                        nn.init.eye_(self.weight.data)

                    setattr(
                        codebook_transform, "initialize", types.MethodType(initialize, codebook_transform)
                    )
            self.code_transform = codebook_transform

        self.learnable_codebook = learnable_codebook and not self.ema_update
        self.embed = nn.Parameter(embed, requires_grad=self.learnable_codebook)
        self.register_buffer("embed_avg", embed.clone())

    def _accumulate_input_batch(self, x):
        num_samples_remain = self._init_input_samples - self._init_input_count
        if x.shape[0] > num_samples_remain:
            x = x[:num_samples_remain]
        x = x.detach().clone()
        self._init_input_batches.append(x)
        self._init_input_count += x.shape[0]

        if self._init_input_count < self._init_input_samples:
            return  # wait for more inputs

        inputs = torch.cat(self._init_input_batches, dim=0)  # (num_samples, dim_feature)
        # free up memory
        del self._init_input_samples
        del self._init_input_batches
        del self._init_input_count
        self._init_embed(inputs)

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

    @torch.no_grad()
    def _expire_codes(self, inputs) -> Tensor | None:
        if self.threshold_ema_dead_code == 0:
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

        return num_expired_codes

    @torch.no_grad()
    def _update_ema(self, x, embed_indices, cluster_size):
        cluster_size = cluster_size.float()
        ema_inplace(self.cluster_size, cluster_size, self.ema_decay)

        if self.ema_update:
            embed_sum = torch.zeros_like(self.embed_avg)  # (codebook_size, dim_feature)
            embed_sum.scatter_add_(0, embed_indices[:, None].expand(-1, self.dim_feature), x)

            reduce(embed_sum, reduction="sum")  # (codebook_size, dim_feature)
            ema_inplace(self.embed_avg, embed_sum, self.ema_decay)

            def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
                x_sum = x.sum(dim=dim, keepdim=True)
                return x_sum * (x + eps) / (x_sum + n_categories * eps)

            cluster_size_smoothed = laplace_smoothing(self.cluster_size, self.codebook_size)
            embed_normalized = self.embed_avg / cluster_size_smoothed[:, None]
            if self.use_cosine_sim:
                embed_normalized = l2_norm(embed_normalized)
            self.embed.data.copy_(embed_normalized)

    def _loss(self, x, quantized, dists):
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
            entropy_reg_loss = entropy_regularization(-dists, self.entropy_reg_temp)
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

        if not self.inited:
            self._accumulate_input_batch(x)
            return x, None

        # expire codes if we are in training mode
        num_expired_codes = None
        if self.training:
            num_expired_codes = self._expire_codes(x)
        if num_expired_codes is None:
            num_expired_codes = torch.zeros((), device=x.device)

        # get codebook embeddings
        codebook = self.codebook  # (codebook_size, dim_feature)

        # compute distances to codebook embeddings
        if self.use_cosine_sim:
            dists = cosine_dist(x, codebook)  # (N, codebook_size)
        else:
            dists = l2_dist(x, codebook)  # (N, codebook_size)

        # find closest codebook embeddings
        embed_indices = gumbel_sample(
            logits=-dists,
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

        # compute perplexity
        cluster_size = torch.bincount(embed_indices.view(-1), minlength=self.codebook_size)
        reduce(cluster_size, reduction="sum")
        perplexity = compute_perplexity(cluster_size)

        # compute VQ losses
        total_loss, loss_terms = self._loss(x, quantized, dists)

        # gather inference and loss statistics
        info = {
            "dists": dists,
            "embed_indices": embed_indices,
            "perplexity": perplexity,
            "normalized_perplexity": perplexity / self.codebook_size,
            "num_expired_codes": num_expired_codes,
            "loss": total_loss,
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
            mean_attributes += [key for key in raw_info.keys() if key.startswith("loss")]

            aggregated_info = {key: list_mean(raw_info[key]) for key in mean_attributes}
            aggregated_info["embed_indices"] = torch.stack(
                raw_info["embed_indices"], dim=1
            )  # (N, num_codebooks)
            aggregated_info["num_expired_codes"] = torch.sum(
                torch.stack([info["num_expired_codes"] for info in info_groups])
            )

        return x_q, aggregated_info, raw_info
