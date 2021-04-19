import math
import os

import numpy as np
import pandas as pd
import pickle as pk

from ..utils import dump
from ..optim import cost_model
from tqdm import tqdm

from functools import reduce


###
# Profiling Setup
###


RED = 0.000001 # preventing overflow


def prepare_configs_comb(cs, ks, ims, ss, fs, max_size):
    layer = [(c, k, im, s, f) for f in fs for s in ss for im in ims for k in ks for c in cs if s <= f and f <= im]
    layer = list(filter(lambda x: filter_max_size(x, max_size), layer))

    transform = list({(c, im) for (c, k, im, s, f) in layer})

    return layer, transform


def get_transform_configs(layer_configs):
    return list({(c, im) for (c, k, im, s, f) in layer_configs})


def get_configs_for(models):
    configs = [cost_model.get_configs_of(model, size)[0] for model, size in models]
    configs = reduce(lambda acc, x: acc + list(x), configs, [])
    
    return list(set(map(tuple, configs)))


def dump_configs(layer, transform, dir, bs=12500):
    write_file(os.path.join(dir, "transforms"), transform, write_transform)

    for i, l in enumerate(split_in_batches(layer, bs)):
        write_file(os.path.join(dir, f"layer_part_{i}"), l, write_scenario)


def split_in_batches(configs, bs=12500):
    batches = []
    for i in range(len(configs) // bs + 1):
        batches.append(configs[bs * i:bs * (i + 1)])
    return batches


def layer_to_size(layer):
    c, k, im, s, f = layer
    return math.ceil((RED*f*f*k*c*im*im)/(s*s))


def filter_max_size(layer, max_size):
    return layer_to_size(layer) < max_size


def write_file(path, params, param_writer):
    with open(path, "w") as file:
        file.write("declare -A SCENARIO_PARAMETERS\n")
        file.write("SCENARIO_PARAMETERS=(\n")
        param_writer(params, file)
        file.write(")\n")


def write_scenario(params, file):
    for i in range(len(params)):
        c, k, im, s, f = params[i]
        file.write(f'\t[{i}]="{k} {c} {s} {im} {im} {f} {f} 0"\n')
        

def write_transform(params, file):
    for i in range(len(params)):
        c, im = params[i]
        file.write(f'\t[{i}]="8 {c} 1 {im} {im} 3 3 0"\n')


###
# Result Processing
###


def process_results(scenario_file, results_dir, primitives, ignore_first=10, dump_at=None, is_transform=False):
    results, results_summary = read_results(results_dir, primitives, ignore_first)
    scenario = read_scenario(scenario_file, is_transform)

    df = make_dataframe(results, scenario)
    df_summary = make_dataframe(results_summary, scenario)

    dump((df, df_summary), dump_at)

    return df, df_summary


def combine_dataframe_parts(df_dir, dump_at=None):
    parts = os.listdir(df_dir)
    parts = [pk.load(open(os.path.join(df_dir, part), "rb")) for part in parts]

    df = pd.concat([part[0] for part in parts], ignore_index=True)
    df_summary = pd.concat([part[1] for part in parts], ignore_index=True)
    
    dump((df, df_summary), dump_at)

    return df, df_summary


def merge_prims_and_transforms(df_prim, df_trans, dump_at=None):
    prims = list(filter(lambda x: len(x) > 2, df_prim.keys()))
    transforms = list(filter(lambda x: len(x) > 2, df_trans.keys()))

    prims_values = df_prim[prims].values
    prims_vars = df_prim[["c", "k", "im", "f", "s"]].values

    result = []

    for i in tqdm(range(len(df_prim))):
        c, _, im, _, _ = prims_vars[i]
        
        df_sub = df_trans
        df_sub = df_sub[df_sub["c"] == c]
        df_sub = df_sub[df_sub["im"] == im]

        layer_vals = prims_values[i:i+1]
        trans_vals = df_sub[transforms].values

        result.append(np.concatenate([prims_vars[i:i+1], trans_vals, layer_vals], axis=1))

    result = np.array(result).reshape(-1, len(prims) + len(transforms) + 5)
    df = pd.DataFrame(result, columns=["c", "k", "im", "f", "s"] + transforms + prims)

    dump(df, dump_at)

    return df


def make_dataframe(results, scenarios):
    for r, s in zip(results, scenarios):
        r.update(s)

    return pd.DataFrame(data=results)


def read_results(results_dir, primitives, ignore_first=10):
    files = range(len(os.listdir(results_dir)))

    results = []
    results_summary = []

    for file in files:
        result = {prim: get_primitive_runtimes(results_dir, file, prim) for prim in primitives}

        results.append(result)
        results_summary.append({prim: np.median(value[ignore_first:]) for prim, value in result.items()})

    return results, results_summary


def read_scenario(file, is_transform=False):
    with open(file) as f:
        scenarios = f.readlines()
    
    scenarios = scenarios[2:-1]
    scenarios = list(map(lambda x: read_scenario_line(x, is_transform), scenarios))
    scenarios = np.array(scenarios)
    
    return scenarios

    
def get_primitive_runtimes(results_dir, ith, prim):    
    path = results_dir + str(ith) + "/" + prim + "/hns.dat"
    
    if os.path.exists(path):
        with open(results_dir + str(ith) + "/" + prim + "/hns.dat") as file:
            return list(map(lambda x: int(float(x.strip())), file.readlines()))
    else:
        return [0] * 25


def read_scenario_line(line, is_transform=False):
    args = list(map(int, line[line.index('"') + 1:-2].split(" ")))
    k, c, s, im, im, f, f, _ = args
    
    if not is_transform:
        return { "k": k, "c": c, "f": f, "s": s, "im": im }
    else:
        return { "c": c, "im": im }
