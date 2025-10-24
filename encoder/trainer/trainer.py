import os
import random
import numpy as np
from numpy import random
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.optim as optim
from trainer.metrics import Metric
from models.bulid_model import build_model
from config.configurator import configs


def init_seed():
    if (seed := configs['train']['seed']):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            if configs['stage']:
                self.optimizer = optim.Adam(# model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
                [
                    {"params": model.vqraf.parameters(), "lr": optim_config['lr']},
                    {"params": list(set(model.parameters()) - set(model.vqraf.parameters())),
                     "lr": optim_config['lr'] / 10, "weight_decay": 2 * model.hyper_config['reg_weight']},
                ])
            else:
                self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay= 2 * model.hyper_config['reg_weight'])
                

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        # start this epoch
        model.train()
        for i, tem in tqdm(enumerate(train_dataloader), desc=f'[Epoch {epoch_idx}]', total=len(train_dataloader)):
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            self.optimizer.zero_grad()

            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            self.logger.log_loss(loss_dict, data_type='Batch')

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        self.logger.log_loss(loss_log_dict, data_type='Epoch')


    def train(self, model):
        now_patience = 0
        best_epoch = 0
        best_recall = -1e9
        self.create_optimizer(model)
        train_config = configs['train']
        for epoch_idx in range(train_config['epoch']):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result = self.evaluate(model)

                if eval_result['recall'][-1] > best_recall:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_recall = eval_result['recall'][-1]
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    now_patience += 1

                # early stop
                if now_patience == configs['train']['patience']:
                    break

        # evaluation again
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        self.evaluate(model)

        # final test
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        test_result = self.test(model)

        # save result
        self.save_model(model)
        print("Best Epoch {}. Final test result: {}.".format(best_epoch, test_result))

    def evaluate(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Valid')
        return eval_result

    def test(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Valid')
        return eval_result
    
    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            save_dir_path = './encoder/checkpoint/{}'.format(model_name)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)

            if configs["stage"] == "map":
                save_model_path = f"./encoder/checkpoint/{model_name}/{model_name}-{configs['data']['name']}-{configs['train']['seed']}_map.pth"
            else:
                save_model_path = f"./encoder/checkpoint/{model_name}/{model_name}-{configs['data']['name']}-{configs['train']['seed']}.pth"
            torch.save(model_state_dict, save_model_path)
            print("Save model parameters to {}".format(save_model_path))


    def load_model(self, model, pretrain_path):
            model.load_state_dict(torch.load(pretrain_path))
            print("Load model parameters from {}".format(pretrain_path))
            
