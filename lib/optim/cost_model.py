import torch
import numpy as np

from ..ml.data_proc import denorm, normalize
from ..ds.cleaning import clean_dict
from ..utils import dump


def build_cost_model_of(target, prims, model, im_size, device, denorm_data, dump_at=None, arm=False):
    configs, names = get_configs_of(target, im_size, for_arm=arm)
    return build_cost_model(configs, prims, model, device, denorm_data, dump_at), names


def build_true_cost_model_of(target, prims, df, im_size, dump_at=None, trans=False):
    configs, names = get_configs_of(target, im_size)
    return build_true_cost_model(configs, prims, df, dump_at, trans=trans), names


def build_cost_model(configs, prims, model, device, denorm_data, dump_at=None):
    ym, ys, xm, xs = denorm_data
    
    preds = denorm(model(normalize(configs, xm, xs).to(device)), ym, ys)
    preds = list(map(lambda x: dict(zip(prims, x)), preds))
    
    for v in list(zip(configs[:,3:5], preds)):
        config, res = v
        
        if len(config) == 1:
            config = [1, config[0]]
        
        clean_dict(*config, res)

    dump(preds, dump_at)
    return preds


def build_true_cost_model(configs, prims, df, dump_at=None, trans=False):
    res = []
    
    for config in configs:
        c, k, im, s, f = config
        
        part = df
        part = part[part["c"] == c]
        part = part[part["im"] == im]
        
        if trans == False:
            part = part[part["k"] == k]
            part = part[part["s"] == s]
            part = part[part["f"] == f]
           
        part = part[prims]
        
        if len(part) == 0:
            res.append(None)
        else:
            res.append(dict(part.iloc[0]))

    dump(res, dump_at)
    return res


def get_configs_of(target, im_size, for_arm=False):
    replace_model_layers(target, torch.nn.Conv2d, lambda x: Conv2dTracer(x))
    _ = target(torch.randn((2, 3, im_size, im_size)))
    
    return read_layer_params(target, Conv2dTracer, for_arm=for_arm)


def replace_model_layers(model, kind, f):
    for name, mod in list(model.named_children()):
        if isinstance(mod, kind):
            setattr(model, name, f(mod))
        else:
            replace_model_layers(mod, kind, f)


def read_layer_params(model, kind, prefix="", for_arm=False):
    res = []
    names = []
    
    for name, mod in list(model.named_children()):
        if isinstance(mod, kind):
            if for_arm:
                res.append([mod.c, mod.k, mod.im, mod.f])
            else:
                res.append([mod.c, mod.k, mod.im, mod.s, mod.f])
            names.append(prefix + "." + name)
        else:  
            newres, newnames = read_layer_params(mod, kind, prefix + "." + name, for_arm)
            
            res.extend(newres)
            names.extend(newnames)
    
    return np.array(res), names


class Conv2dTracer(torch.nn.Module):
    def __init__(self, module):
        super(Conv2dTracer, self).__init__()
        
        self.mod = module
        
        self.im = None
        self.f = module.kernel_size[0]
        self.s = module.stride[0]
        self.c = module.in_channels
        self.k = module.out_channels
    
    def forward(self, x):
        self.im = x.size(3)
        return self.mod(x)