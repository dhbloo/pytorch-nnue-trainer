import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import PartialState
from accelerate.utils import gather, reduce
from pykeops.torch import LazyTensor


def l2_dist(x: torch.Tensor, y: torch.Tensor) -> LazyTensor:
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


def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> LazyTensor:
    """
    Compute the cosine similarity between x and y.
    Args:
        x: Input embedding tensor of shape (N, dim_feature).
        y: Codebook tensor of shape (M, dim_feature).
    Returns:
        dists: A tensor of shape (N, M).
    """
    x = F.normalize(x, p=2, dim=-1, eps=1e-6)
    y = F.normalize(y, p=2, dim=-1, eps=1e-6)
    x_i = LazyTensor(x[:, None, :])  # (N, 1, dim_feature)
    y_j = LazyTensor(y[None, :, :])  # (1, M, dim_feature)
    dists = 1 - (x_i * y_j).sum(-1)  # (N, M)
    return dists


def uniform_init(*shape) -> torch.Tensor:
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


def gumbel_sample(
    logits: LazyTensor,
    temperature = 1.,
    stochastic = False,
    softmax_gradient = False,
    training = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from a categorical distribution using the Gumbel-Softmax trick.
    Args:
        logits: Unnormalized log probabilities of shape (N, num_classes).
        temperature: Temperature parameter for sampling.
        stochastic: Whether to use Gumbel sampling.
        softmax_gradient: Whether to use the Softmax gradients.
        training: Whether the model is in training mode.
    Returns:
        ind: Indices of the sampled elements.
        one_hot: One-hot tensor (N, num_classes) of the sampled elements.
    """
    if training and stochastic and temperature > 0:
        gumbel_noise = uniform_lazy_tensor_2d(*logits.shape, device=logits.variables[0].device)
        gumbel_noise = -(-(gumbel_noise * (1-2e-7) + 1e-7).log()).log()
        sampling_logits = (logits / temperature) + gumbel_noise
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=1).squeeze(1)  # (N,)
    one_hot = F.one_hot(ind, logits.shape[-1])  # (N, num_classes)

    if not softmax_gradient or temperature <= 0 or not training:
        return ind, one_hot

    π1 = (logits / temperature).softmax(dim=-1)
    one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot


def compute_perplexity(cluster_size: torch.Tensor) -> torch.Tensor:
    probs = cluster_size / cluster_size.sum()
    entropy = probs * torch.log(probs + 1e-10)
    perplexity = torch.exp(-torch.sum(entropy))
    return perplexity


def entropy_regularization(logits: LazyTensor, temperature = 1.) -> torch.Tensor:
    logits = logits / temperature  # (N, codebook_size)
    logits_max_and_exp = logits.reduction('Max_SumShiftExp', dim=1)  # (N, 2)
    logits_max = logits_max_and_exp[:, 0].contiguous()
    logits_sumexp = logits_max_and_exp[:, 1].contiguous()
    logits_exp = (logits - LazyTensor(logits_max[:, None], axis=0)).exp()  # (N, codebook_size)
    softprobs = logits_exp / LazyTensor(logits_sumexp[:, None], axis=0)  # (N, codebook_size)
    avg_probs = (softprobs.sum(0) / logits.shape[0]).squeeze(1)  # (codebook_size,)
    return -(-avg_probs * torch.log(avg_probs.clamp_min(1e-5))).sum(dim=-1).mean()


def sample_vectors(inputs: torch.Tensor, num_samples: int) -> torch.Tensor:
    num_inputs, device = inputs.shape[0], inputs.device
    assert num_inputs > 0, "Can not sample empty input tensor"
    if num_inputs >= num_samples:
        indices = torch.randperm(num_inputs, device=device)[:num_samples]
    else:
        indices = torch.randint(0, num_inputs, (num_samples,), device=device)
    return inputs[indices]


def kmeans(
    inputs: torch.Tensor, 
    num_clusters: int, 
    num_iters: int = 10, 
    use_cosine_sim: bool = False,
) -> torch.Tensor:
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
    num_processes, process_idx = PartialState().num_processes, PartialState().process_index
    num_samples = num_samples_per_rank * num_processes
    if num_samples < num_clusters:
        raise ValueError(f"Number of samples {num_samples} < number of clusters {num_clusters}")

    # initialize cluster centroids
    num_sample_clusters, num_sample_clusters_remain = divmod(num_clusters, num_processes)
    if process_idx < num_sample_clusters_remain:
        num_sample_clusters += 1
    means = sample_vectors(inputs, num_sample_clusters)  # (num_sample_clusters, dim_feature)
    means : torch.Tensor = gather(means)  # (num_clusters, dim_feature)

    # perform K-means iterations
    for _ in range(num_iters):
        if use_cosine_sim:
            dist = cosine_dist(inputs, means)  # (num_samples_per_rank, num_clusters)
        else:
            dist = l2_dist(inputs, means)  # (num_samples_per_rank, num_clusters)

        buckets = dist.argmin(dim=1).squeeze(1)  # (num_samples_per_rank,)
        bins = torch.bincount(buckets, minlength=num_clusters)  # (num_clusters,)
        bins = reduce(bins, reduction='sum')  # (num_clusters,)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = torch.zeros(num_clusters, dim_feature, device=device, dtype=dtype)
        new_means.scatter_add_(0, buckets[:, None].expand(-1, dim_feature), inputs)
        new_means = reduce(new_means, reduction='sum')  # (num_clusters, dim_feature)
        new_means = new_means / bins_min_clamped[:, None]  # (num_clusters, dim_feature)

        if use_cosine_sim:
            new_means = F.normalize(new_means, p=2, dim=-1, eps=1e-6)

        means = torch.where(zero_mask[:, None], means, new_means)

    return means, bins


def ema_inplace(tensor : torch.Tensor, new : torch.Tensor, decay: float) -> torch.Tensor:
    return tensor.lerp_(new, 1 - decay)


class VectorQuantize(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Args:
        codebook_size: The number of embeddings in the codebook.
        dim_feature: Dimension of the embeddings.
        beta: commitment cost used in loss term, beta * ||x_q-sg(x)||^2
    """
    def __init__(self, 
                 codebook_size: int, 
                 dim_feature: int, 
                 kmeans_init=False,
                 kmeans_sample_multiplier=1,
                 kmeans_iter=10,
                 ema_decay=0.9,
                 commitment_weight=0.25,
                 learnable_codebook=True,
                 sampling_temp=1.0,
                 stochastic_sampling=False,
                 softmax_gradient=False,
                 entropy_reg_weight=0.0,
                 entropy_reg_temp=0.01):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim_feature = dim_feature
        self.kmeans_init = kmeans_init
        self.kmeans_sample_multiplier = kmeans_sample_multiplier
        self.kmeans_iter = kmeans_iter
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight
        self.sampling_temp = sampling_temp
        self.stochastic_sampling = stochastic_sampling
        self.softmax_gradient = softmax_gradient
        self.entropy_reg_weight = entropy_reg_weight
        self.entropy_reg_temp = entropy_reg_temp

        self.register_buffer('inited', torch.zeros([], dtype=torch.bool))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        if kmeans_init:
            embed = torch.zeros(codebook_size, dim_feature)
            self.init_input_samples = (codebook_size * kmeans_sample_multiplier // PartialState().num_processes)
            self.init_input_batches = []
            self.init_input_count = 0
        else:
            embed = uniform_init(codebook_size, dim_feature)
            self.inited.data.fill_(True)
        self.embeddings = nn.Parameter(embed, requires_grad=learnable_codebook)

    def _accumulate_input_batch(self, x):
        num_samples_remain = self.init_input_samples - self.init_input_count
        if x.shape[0] > num_samples_remain:
            x = x[:num_samples_remain]
        self.init_input_batches.append(x.detach().clone())
        self.init_input_count += x.shape[0]

        if self.init_input_count < self.init_input_samples:
            return  # wait for more inputs

        inputs = torch.cat(self.init_input_batches, dim=0)  # (num_samples, dim_feature)
        # free up memory
        del self.init_input_samples
        del self.init_input_batches
        del self.init_input_count
        self._init_embed(inputs)

    @torch.no_grad()
    def _init_embed(self, inputs):
        num_total_inputs = inputs.shape[0] * PartialState().num_processes
        assert num_total_inputs >= self.codebook_size, \
            f"Number of inputs {num_total_inputs} < codebook size {self.codebook_size}"
        
        if num_total_inputs > self.codebook_size:
            embed, cluster_size = kmeans(
                inputs=inputs,
                num_clusters=self.codebook_size,
                num_iters=self.kmeans_iter,
                use_cosine_sim=False,
            )

            # Normalize cluster size so that we can assume num_inputs equals to codebook_size
            cluster_size = cluster_size.float()
            cluster_size *= self.codebook_size / cluster_size.sum()
        else:
            embed = gather(inputs)
            cluster_size = torch.ones(self.codebook_size, device=inputs.device)

        self.embeddings.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.fill_(True)

    def update_ema(self, cluster_size):
        cluster_size = cluster_size.float()
        # Normalize cluster size so that we can assume num_inputs equals to codebook_size
        cluster_size.mul_(self.codebook_size / cluster_size.sum())
        ema_inplace(self.cluster_size, cluster_size, self.ema_decay)

    @property
    def codebook(self) -> torch.Tensor:
        return self.embeddings

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.codebook)  # (..., dim_feature)

    def forward(self, x : torch.Tensor):
        """
        Get the quantized version x_q of the continuous input x.
        Args:
            x: A continuous tensor of shape (N, dim_feature).
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
        assert x.shape[-1] == self.dim_feature, \
            f"Wrong input dim {x.shape[-1]} != {self.dim_feature}"
        
        if not self.inited:
            self._accumulate_input_batch(x)
            return x, None
        
        # get codebook embeddings
        codebook = self.codebook  # (codebook_size, dim_feature)

        # compute distances to codebook embeddings
        dists = l2_dist(x, codebook)  # (N, codebook_size)
        
        # find closest codebook embeddings
        embed_indices, embed_onehot = gumbel_sample(
            logits=-dists, 
            temperature=self.sampling_temp, 
            stochastic=self.stochastic_sampling, 
            softmax_gradient=self.softmax_gradient, 
            training=self.training,
        )  # (N,),  (N, codebook_size)

        # get quantized vectors
        if self.training and self.softmax_gradient:
            quantized = torch.matmul(embed_onehot, codebook)  # (N, dim_feature)
        else:
            quantized = self.from_indices(embed_indices)  # (N, dim_feature)

        # preserve gradients
        x_q = x + (quantized - x).detach()

        # compute perplexity
        cluster_size = torch.bincount(embed_indices.view(-1), minlength=self.codebook_size)
        reduce(cluster_size, reduction='sum')
        perplexity = compute_perplexity(cluster_size)

        info = {
            'dists': dists,
            'embed_indices': embed_indices,
            'embed_onehot': embed_onehot,
            'perplexity': perplexity,
            'normalized_perplexity': perplexity / self.codebook_size,
        }

        # perform EMA update
        if self.training:
            self.update_ema(cluster_size)

        # compute VQ-VAE loss
        loss_embed = (quantized - x.detach()).square().mean()
        loss_commitment = (quantized.detach() - x).square().mean()
        loss = loss_embed + self.commitment_weight * loss_commitment

        # add entropy regularization loss
        if self.entropy_reg_weight > 0:
            entropy_reg_loss = entropy_regularization(-dists, self.entropy_reg_temp)
            loss = loss + self.entropy_reg_weight * entropy_reg_loss
        else:
            entropy_reg_loss = torch.tensor(0.0, device=x.device)

        info.update({
            'loss': loss,
            'loss_embed': loss_embed,
            'loss_commitment': loss_commitment,
            'loss_entropy': entropy_reg_loss,
        })

        return x_q, info

