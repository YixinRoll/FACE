import swanlab


class Logger():
    def __init__(self, configs):
        # Try initializing swanlab a few times in case of transient errors; format project name safely.
        for _ in range(3):
            try:
                swanlab.init(project=f"NIPS_final_{configs['data']['name']}_{configs['model']['name']}", config={**configs['optimizer'], **configs['train'], **configs['test'], **configs['data'], **configs['model']}, name=configs['remark'])
                break
            except Exception:
                import time
                time.sleep(1)
        else:
            # Final attempt to allow the exception to propagate if persistent
            swanlab.init(project=f"NIPS_final_{configs['data']['name']}_{configs['model']['name']}", config={**configs['optimizer'], **configs['train'], **configs['test'], **configs['data'], **configs['model']}, name=configs['remark'])
        
    def log_loss(self, loss_log_dict, data_type):
        swanlab.log({f'{data_type}/{key}': value for key, value in loss_log_dict.items()})
        
    def log_eval(self, eval_result, k, data_type):
        message = ''
        for metric in eval_result:
            message += '['
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
                swanlab.log({"{}/{}@{}".format(data_type, metric, k[i]):eval_result[metric][i]})
            message += '] '
        print(message)
