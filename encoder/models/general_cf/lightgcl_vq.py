import torch as t
import numpy as np
from torch import nn
from models.base_model import BaseModel
from config.configurator import configs
from models.aug_utils import SvdDecomposition
from models.loss_utils import cal_bpr_loss, reg_params, cal_align_loss
from vqraf import VQRAF

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform_

class LightGCL_vq(BaseModel):
    def __init__(self, data_handler):
        super(LightGCL_vq, self).__init__(data_handler)

        train_mat = data_handler._load_one_mat(data_handler.trn_file)
        rowD = np.array(train_mat.sum(1)).squeeze()
        colD = np.array(train_mat.sum(0)).squeeze()
        for i in range(len(train_mat.data)):
            train_mat.data[i] = train_mat.data[i] / pow(rowD[train_mat.row[i]] * colD[train_mat.col[i]], 0.5)
        adj_norm = self._scipy_sparse_mat_to_torch_sparse_tensor(train_mat)
        self.adj = adj_norm.coalesce().cuda()

        self.svd_decompose = SvdDecomposition(svd_q=configs['model']['svd_q'])
        self.ut, self.vt, self.u_mul_s, self.v_mul_s = self.svd_decompose(self.adj)

        self.temp = configs['model']['temp']
        self.dropout = configs['model']['dropout']
        self.layer_num = configs['model']['layer_num']
        self.cl_weight = configs['model']['cl_weight']

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        self.act = nn.LeakyReLU(0.5)
        self.Ws = nn.ModuleList([W_contrastive(self.embedding_size) for i in range(self.layer_num)])
        self.is_training = True

        self.usrprf_repre = t.tensor(configs['usrprf_repre']).float().cuda()
        self.itmprf_repre = t.tensor(configs['itmprf_repre']).float().cuda()

        # vq
        self.word_num = self.hyper_config['word_num']
        self.word_dim = self.hyper_config['word_dim']
        self.vq_weight = self.hyper_config['vq_weight']
        self.recons_weight = self.hyper_config['recons_weight']
        self.align_weight = self.hyper_config['align_weight']
        self.vqraf = VQRAF(input_dim=self.embedding_size, word_num=self.word_num, word_dim = self.word_dim, dataset_name = configs['data']['name'], llm_name=configs['llm'])

        if configs["stage"] == "map":
            load_model_name = configs["model"]["name"][:-3]
            load_model_path = f"./encoder/checkpoint/{load_model_name}/{load_model_name}-{configs['data']['name']}-{configs['train']['seed']}.pth"
            self.load_state_dict(t.load(load_model_path), strict=False)
            print(f"Successfully load model from {load_model_path}")
        else:
            load_model_name = configs["model"]["name"]
            load_model_path = f"./encoder/checkpoint/{load_model_name}/{load_model_name}-{configs['data']['name']}-{configs['train']['seed']}_map.pth"
            self.load_state_dict(t.load(load_model_path))
            print(f"Successfully load model from {load_model_path}")

        self.E_u_list = [None] * (self.layer_num+1)
        self.E_i_list = [None] * (self.layer_num+1)
        self.E_u_list[0] = self.user_embeds
        self.E_i_list[0] = self.item_embeds
        self.Z_u_list = [None] * (self.layer_num+1)
        self.Z_i_list = [None] * (self.layer_num+1)
        self.G_u_list = [None] * (self.layer_num+1)
        self.G_i_list = [None] * (self.layer_num+1)
        self.G_u_list[0] = self.user_embeds
        self.G_i_list[0] = self.item_embeds
        self.E_u = None
        self.E_i = None        

    def _scipy_sparse_mat_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data)
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)

    def _spmm(self,sp, emb):
        sp = sp.coalesce()
        cols = sp.indices()[1]
        rows = sp.indices()[0]
        col_segs = emb[cols] * t.unsqueeze(sp.values(),dim=1)
        result = t.zeros((sp.shape[0],emb.shape[1])).cuda()
        result.index_add_(0, rows, col_segs)
        return result

    def _sparse_dropout(self, mat, dropout):
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()
        return t.sparse.FloatTensor(indices, values, size)

    def forward(self, test=False):
        if test and self.E_u is not None:
            return self.E_u, self.E_i
        for layer in range(1, self.layer_num+1):
            # GNN propagation
            self.Z_u_list[layer] = self._spmm(self._sparse_dropout(self.adj,self.dropout), self.E_i_list[layer-1])
            self.Z_i_list[layer] = self._spmm(self._sparse_dropout(self.adj,self.dropout).transpose(0,1), self.E_u_list[layer-1])

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer-1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer-1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]  # + self.E_u_list[layer-1]
            self.E_i_list[layer] = self.Z_i_list[layer]  # + self.E_i_list[layer-1]

        # aggregate across layers
        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        return self.E_u, self.E_i
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
    
    def _pick_prfs(self, user_prfs, item_prfs, batch_data):
        ancs, poss, negs = batch_data
        anc_prfs = [user_prfs[anc.item()] for anc in ancs]
        pos_prfs = [item_prfs[pos.item()] for pos in poss]
        neg_prfs = [item_prfs[neg.item()] for neg in negs]
        return anc_prfs, pos_prfs, neg_prfs


    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        # do vq
        entity_embeds = t.cat([anc_embeds, pos_embeds, neg_embeds], dim=0)
        entity_embeds_vq, vq_loss, recons_loss, colla_repre = self.vqraf(entity_embeds, configs["stage"])

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0] 
        # get the semantic representations
        ancprf_repre, posprf_repre, negprf_repre = self._pick_embeds(self.usrprf_repre, self.itmprf_repre, batch_data)
        semantic_repre = t.cat([ancprf_repre, posprf_repre, negprf_repre], dim=0)
        align_loss = cal_align_loss(colla_repre, semantic_repre) 

        # pos_scores = (anc_embeds * pos_embeds).sum(-1)
        # neg_scores = (anc_embeds * neg_embeds).sum(-1)
        # bpr_loss = -(pos_scores - neg_scores).sigmoid().log().mean()
        G_u_norm = self.G_u
        E_u_norm = self.E_u
        G_i_norm = self.G_i
        E_i_norm = self.E_i
        neg_score = t.log(t.exp(G_u_norm[ancs] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += t.log(t.exp(G_i_norm[poss] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (t.clamp((G_u_norm[ancs] * E_u_norm[ancs]).sum(1) / self.temp, -5.0, 5.0)).mean() + \
                    (t.clamp((G_i_norm[poss] * E_i_norm[poss]).sum(1) / self.temp, -5.0, 5.0)).mean()
        cl_loss = -pos_score + neg_score

        loss = bpr_loss + self.cl_weight * cl_loss + self.vq_weight * vq_loss + self.recons_weight * recons_loss + self.align_weight * align_loss
        losses = {'bpr_loss': bpr_loss, 'cl_loss': cl_loss, 'vq_loss': vq_loss, 'recons_loss': recons_loss, 'align_loss': align_loss}
        return loss, losses
    
    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(test=True)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    

class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(t.empty(d,d)))

    def forward(self,x):
        return x @ self.W