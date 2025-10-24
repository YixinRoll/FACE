import torch as t
import torch.nn.functional as F
import swanlab

def cal_align_loss(A, B, temperature=0.02):
    if A is None:
        return t.tensor(0.0, device=B.device)
    sim_matrix = A @ B.T
    # swanlab.log({"diag_avg": t.diag(sim_matrix).mean(), 
    #            "non_diag_avg": (sim_matrix.sum() - t.diag(sim_matrix).sum()) / (sim_matrix.shape[0] * sim_matrix.shape[1] - sim_matrix.shape[0])})
    sim_matrix = sim_matrix / temperature
    labels = t.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# sim_matrix = A @ B.T / temperature
# sim_matrix_scaled = t.exp(sim_matrix)
# loss = - t.mean(t.log(t.diag(sim_matrix_scaled) / t.sum(sim_matrix_scaled, dim=1)))

# def contrastive_loss(logits):
#     return F.cross_entropy(logits, t.arange(len(logits), device=logits.device))

# def cal_align_loss(A, B, temperature=0.02):
#     scaled_sim_matrix = t.matmul(A, B.T) / temperature
#     loss = (contrastive_loss(scaled_sim_matrix) + contrastive_loss(scaled_sim_matrix.T))/2
#     return loss




def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))

def reg_pick_embeds(embeds_list):
    reg_loss = 0
    for embeds in embeds_list:
        reg_loss += embeds.square().sum()
    return reg_loss

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss

def reg_params(model):
    reg_loss = 0
    for W in [model.user_embeds, model.item_embeds]:
        reg_loss += W.norm(2).square()
    return reg_loss

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def ssl_con_loss(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()

def alignment(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return t.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()