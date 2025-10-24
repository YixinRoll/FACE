import torch as t
import torch.nn as nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform_

class GMF(BaseModel):
    def __init__(self, data_handler):
        super(GMF, self).__init__(data_handler)
        # model parameters
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))


    def forward(self):
        return self.user_embeds, self.item_embeds
    
    def cal_loss(self, batch_data):
        user_embeds, item_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        loss = bpr_loss
        losses = {'bpr_loss': bpr_loss} # reg loss in optimizer
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    