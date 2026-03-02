import torch
from .base import BaseLoss, gather_and_scale_wrapper


def compute_hash_loss(z, batch_size, alpha=0.1, lamda=0.1):
    P, K = batch_size
    m = 0.2 * z.size(1) ** 0.5

    indices = torch.arange(z.size(0))
    pos_indices = indices.view(P, K).roll(shifts=-1, dims=1).reshape(-1)
    neg_indices = indices.view(P, K).roll(shifts=-1, dims=0).reshape(-1)

    ori, pos, neg = z, z[pos_indices], z[neg_indices]
    dist_ap = torch.norm(ori - pos, p=2, dim=1)
    dist_an = torch.norm(ori - neg, p=2, dim=1)

    pair_loss = torch.relu(dist_ap - dist_an + m).mean()
    quant_loss = torch.mean((torch.abs(z) - 1.0) ** 2)
    p_reg = torch.mean(torch.relu(torch.abs(z) - 0.8))

    return pair_loss + alpha * p_reg + lamda * quant_loss


class HashLoss(BaseLoss):
    @gather_and_scale_wrapper
    def forward(self, logits, labels=None):
        loss = compute_hash_loss(logits, [8, 4])
        self.info.update({"loss": loss.detach().clone()})
        return loss, self.info
