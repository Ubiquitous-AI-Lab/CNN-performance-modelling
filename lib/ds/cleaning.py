from .prims import PRIMITIVES, KN2, WINO3, WINO5, IM2, CONV1
from ..utils import dump

from tqdm import tqdm_notebook as tqdm

import numpy as np

RULES = [
    (lambda s, f: s == 1, PRIMITIVES[KN2 + WINO3 + WINO5 + IM2[4:12] + IM2[16:20]]),
    (lambda s, f: f == 3, PRIMITIVES[WINO3]),
    (lambda s, f: f == 5, PRIMITIVES[WINO5]),
    (lambda s, f: f == 1, PRIMITIVES[CONV1])
]

RULES_NP = [
    (lambda data: data[:,0] == 1, PRIMITIVES[KN2 + WINO3 + WINO5 + IM2[4:12] + IM2[16:20]]),
    (lambda data: data[:,1] == 3, PRIMITIVES[WINO3]),
    (lambda data: data[:,1] == 5, PRIMITIVES[WINO5]),
    (lambda data: data[:,1] == 1, PRIMITIVES[CONV1])
]


def clean_dataframe(df, z=0, dump_at=None):
    vrs = df[["s", "f"]].values
    for predicate, prims in RULES_NP:
        mask = list(np.where(~predicate(vrs))[0])
        for p in prims:
            df.loc[mask,p] = z

    dump(df, dump_at)
    return df


def clean_dict(s, f, data):
    layers = data.keys()

    for predicate, prims in RULES:
        if not predicate(s, f):
            for p in prims:
                if p in layers:
                    data[p] = 0
    
    return data


def drop_primitives(df, prims, dump_at=None):
    df = df.drop(columns=prims)
    dump(df, dump_at)

    return df
