import os
import yaml
import pickle
import argparse

def parse_configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lightgcn_vq', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Random Seed')
    parser.add_argument('--cuda', type=int, default=0, help='Device number')
    parser.add_argument('--remark', type=str, default=None, help='Remark for logger')
    parser.add_argument('--llm', type=str, default='miniLM', help='LLM name', choices=['llama2', 'miniLM', 'gpt', 'qwen2'])
    parser.add_argument('--stage', type=str, default=None, choices=['map', 'align'], help="Current stage of FACE")
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda' and args.cuda >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    # model name
    model_name = args.model.lower()

    # read yml file
    with open(f"./encoder/config/modelconf/{model_name}.yml", encoding='utf-8') as f:
        configs = yaml.safe_load(f.read())
        
    configs['model']['name'] = model_name
    configs['data']['name'] = args.dataset
    configs['device'] = args.device            
    configs['train']['seed'] = args.seed

    if args.stage is not None: # for FACE only
        configs['stage'] = args.stage
    else:
        configs['stage'] = None
    
    # semantic embeddings for RLMRec
    usrprf_embeds_path = "./data/{}/usr_emb_np.pkl".format(configs['data']['name'])
    itmprf_embeds_path = "./data/{}/itm_emb_np.pkl".format(configs['data']['name'])
    with open(usrprf_embeds_path, 'rb') as f:
        configs['usrprf_embeds'] = pickle.load(f)
    with open(itmprf_embeds_path, 'rb') as f:
        configs['itmprf_embeds'] = pickle.load(f)

    # semantic representations for VQRAF
    usrprf_repre_path = f"./data/{configs['data']['name']}/usr_repre_np_{args.llm}.pkl"
    itmprf_repre_path = f"./data/{configs['data']['name']}/itm_repre_np_{args.llm}.pkl"
    with open(usrprf_repre_path, 'rb') as f:
        configs['usrprf_repre'] = pickle.load(f)
    with open(itmprf_repre_path, 'rb') as f:
        configs['itmprf_repre'] = pickle.load(f)

    
    usrprf_path = "./data/{}/usr_prf.pkl".format(configs['data']['name'])
    itmprf_path = "./data/{}/itm_prf.pkl".format(configs['data']['name'])
    with open(usrprf_path, 'rb') as f:
        usrprf = pickle.load(f)
        configs['usrprf'] = {k: v['profile'] for k, v in usrprf.items()}
    with open(itmprf_path, 'rb') as f:
        itmprf = pickle.load(f)
        configs['itmprf'] = {k: v['profile'] for k, v in itmprf.items()}

    configs['remark'] = args.remark
    configs['llm'] = args.llm

    return configs

configs = parse_configure()
