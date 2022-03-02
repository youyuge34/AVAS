"""
Torch 训练辅助代码
"""
import torch
from torch import nn
import os, signal, subprocess
import logging

logger = logging.getLogger(__name__)

def all_child(module, classKeyword):
    for name ,child in (module.named_modules()):
        if str(type(child)).find(classKeyword) >= 0:
            yield name, child

def check_gpu():
    gpu_count = 0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info("gpu count: {}".format(gpu_count))
        logger.info(torch.cuda.get_device_name(0))
    else:
        logger.info("No gpu")
    return gpu_count


def move_to(obj, device):
    """
    将整个容器里的 tensor 都移动到对应 device 里
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        return obj


def soft_cross_entropy(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim = 1)
    return -(target * logprobs).sum() / input.shape[0]


def tensor_to_cpu_arr(tensor):
    return tensor.detach().cpu().numpy()


def track_all_tensors(min_occur=5):
    """
    打印当前 GC 对象，用于检测 GPU 显存泄漏问题
    """
    import gc
    cnt = dict()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                sz_key = repr(obj.size())
                if sz_key not in cnt:
                    cnt[sz_key] = 1
                else:
                    cnt[sz_key] += 1
        except:
            pass
    lst = list(cnt.items())
    lst.sort(key=lambda x: x[1], reverse=True)
    for k, v in lst:
        if v >= min_occur:
            print(k, v)


def clone_model(model):
    import copy
    if hasattr(model, 'module'):
        model_clone = copy.deepcopy(model.module)
    else:
        model_clone = copy.deepcopy(model)
    return model_clone


def fix_random_seed(seed=1024):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_model_and_training_state(path, min_loss, epoch, model):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if hasattr(model, 'module'):
        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'state_dict': model.module.state_dict()
        }, path)
    else:
        if hasattr(model, 'train_config'):
            train_config = model.train_config
        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'state_dict': model.state_dict()
        }, path)
    print('save model to', path)


def load_model_and_training_state(path, model, device='cuda'):
    ckpt = torch.load(path, map_location=torch.device(device))
    if hasattr(model, 'module'):
        model.module.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    min_loss = ckpt['min_loss']
    return min_loss


def score_to_predictions(outputs):
    outputs = nn.functional.softmax(outputs, dim=1)
    scores = outputs.max(dim=1, keepdim=True)
    max_idx = scores.indices
    scores = scores.values
    pred = tensor_to_cpu_arr(torch.flatten(max_idx))
    scores = tensor_to_cpu_arr(torch.flatten(scores))
    return pred, scores


def kill_child_processes():
    parent_id = os.getpid()
    ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % parent_id, shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read().decode('utf-8')
    retcode = ps_command.wait()
    for pid_str in ps_output.strip().split("\n")[:-1]:
        try:
            os.kill(int(pid_str), signal.SIGTERM)
        except:
            pass

def random_name(N):
    import random
    import string
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


class Dict2Obj(object):
    def __init__(self, dic):
        self.dic = dic
        for key, val in dic.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [Dict2Obj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, Dict2Obj(val) if isinstance(val, dict) else val)

    def __getitem__(self, key):
        return self.dic[key]
    
    def __setitem__(self, key, val):
        self.dic[key] = val
        if isinstance(val, (list, tuple)):
            setattr(self, key, [Dict2Obj(x) if isinstance(x, dict) else x for x in val])
        else:
            setattr(self, key, Dict2Obj(val) if isinstance(val, dict) else val)

    def __len__(self):
        return len(self.dic)
