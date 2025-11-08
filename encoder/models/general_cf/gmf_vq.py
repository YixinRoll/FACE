import torch as t
from FACE import FACE
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, cal_align_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform_

class GMF_vq(BaseModel):
    def __init__(self, data_handler):
        super(GMF_vq, self).__init__(data_handler)
        # model parameters
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))


        self.usrprf_repre = t.tensor(configs['usrprf_repre']).float().cuda()
        self.itmprf_repre = t.tensor(configs['itmprf_repre']).float().cuda()
        # vq
        self.word_num = self.hyper_config['word_num']
        self.word_dim = self.hyper_config['word_dim']
        self.vq_weight = self.hyper_config['vq_weight']
        self.recons_weight = self.hyper_config['recons_weight']
        self.align_weight = self.hyper_config['align_weight']
        self.vqraf = FACE(input_dim=self.embedding_size, word_num=self.word_num, word_dim = self.word_dim, dataset_name = configs['data']['name'], llm_name=configs['llm'])

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

    def forward(self):
        return self.user_embeds, self.item_embeds

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
        user_embeds, item_embeds = self.forward()
        
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)
        
        entity_embeds = t.cat([anc_embeds, pos_embeds, neg_embeds], dim=0)
        entity_embeds_vq, vq_loss, recons_loss, colla_repre = self.vqraf(entity_embeds, configs["stage"])

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0] 
        # get the semantic representations
        ancprf_repre, posprf_repre, negprf_repre = self._pick_embeds(self.usrprf_repre, self.itmprf_repre, batch_data)
        semantic_repre = t.cat([ancprf_repre, posprf_repre, negprf_repre], dim=0)
        align_loss = cal_align_loss(colla_repre, semantic_repre) 

        loss = bpr_loss + self.vq_weight * vq_loss + self.recons_weight * recons_loss + self.align_weight * align_loss
        losses = {'bpr_loss': bpr_loss, 'vq_loss': vq_loss, 'recons_loss': recons_loss, 'align_loss': align_loss}
        return loss, losses
    
    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    