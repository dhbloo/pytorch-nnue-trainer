import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Args:
        codebook_size: The number of embeddings in the codebook.
        dim_feature: Dimension of the embeddings.
        beta: commitment cost used in loss term, beta * ||z_q-sg(z)||^2
    """
    def __init__(self, codebook_size, dim_feature, beta=0.25, diversity_gamma=0.1, temperature=0.01):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim_feature = dim_feature
        self.beta = beta
        self.diversity_gamma = diversity_gamma
        self.inv_temperature = 1.0 / temperature
        self.embedding = nn.Embedding(codebook_size, dim_feature)

    def forward(self, z : torch.Tensor, beta_multiplier=1):
        """
        Get the quantized version z_q of the continuous input z.
        Args:
            z: A continuous tensor of shape (..., dim_feature).
            beta_multiplier: The weight multiplier on the commitment loss.
        Returns:
            z_q: A quantized tensor of shape (..., dim_feature).
            indices: Indices of the closest embedding, long tensor of shape (...).
            loss: The total VQ-VAE loss including embed and commitment terms, scalar.
            perplexity: The normalized codebook utilization in [0, 1], scalar.
        """
        assert z.shape[-1] == self.dim_feature, f"Wrong z dim {z.shape[-1]} != {self.dim_feature}"

        # compute distances of z and embeddings: (z - e)^2 = z^2 + e^2 - 2 e * z
        # dists = z.square().sum(-1, keepdim=True) + self.embedding.weight.square().sum(-1) \
        #     - 2 * torch.matmul(z, self.embedding.weight.t())  # (..., codebook_size)
        dists = - 2 * torch.matmul(z.detach(), self.embedding.weight.t())  # (..., codebook_size)
        
        # find closest embeddings
        embed_indices = torch.argmin(dists, dim=-1)  # (...,)

        # get quantized vectors
        z_q = self.embedding(embed_indices)  # (..., dim_feature)

        # compute VQ-VAE loss
        loss_embed = (z_q - z.detach()).square().mean()
        loss_commitment = (z_q.detach() - z).square().mean()
        loss = loss_embed + self.beta * beta_multiplier * loss_commitment

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # compute perplexity
        embed_onehot = F.one_hot(embed_indices, self.codebook_size)  # (..., codebook_size)
        embed_onehot = embed_onehot.reshape(-1, self.codebook_size)  # (n, codebook_size)
        probs = embed_onehot.sum(0) / embed_onehot.shape[0]  # (codebook_size)
        entropy = probs * torch.log(probs + 1e-10)  # (codebook_size)
        perplexity = torch.exp(-torch.sum(entropy))

        # add entropy regularization loss
        if self.diversity_gamma > 0:
            softprobs = F.softmax(-dists * self.inv_temperature, dim=-1)  # (..., codebook_size)
            avg_probs = softprobs.reshape(-1, self.codebook_size).mean(0)  # (codebook_size)
            avg_entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
            sample_entropy = -(softprobs * torch.log(softprobs + 1e-10)).sum(-1).mean()
            entropy_loss = sample_entropy - avg_entropy
            loss = loss + self.diversity_gamma * entropy_loss

        return z_q, embed_indices, loss, perplexity

