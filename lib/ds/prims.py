import numpy as np

VARIABLES = np.array(["c", "k", "im", "s", "f"])

TRANSFORMS = np.array([
    'hwc-to-hwc',
    'hwc-to-hcw',
    'hwc-to-chw',
    'hcw-to-hwc',
    'hcw-to-hcw',
    'hcw-to-chw',
    'chw-to-hwc',
    'chw-to-hcw',
    'chw-to-chw',
])

PRIMITIVES = np.array([
    'direct-sum2d',
    'im2col-copy-self-ab-ki',
    'im2col-copy-self-atb-ik',
    'im2col-copy-self-atb-ki',
    'im2col-copy-self-atbt-ik',
    'im2col-copy-short-ab-ki',
    'im2col-copy-short-atb-ik',
    'im2col-copy-short-atb-ki',
    'im2col-copy-short-atbt-ik',
    'im2col-scan-ab-ki',
    'im2col-scan-atb-ik',
    'im2col-scan-atb-ki',
    'im2col-scan-atbt-ik',
    'im2row-copy-short-ab-ik',
    'im2row-copy-short-abt-ik',
    'im2row-copy-short-abt-ki',
    'im2row-copy-short-atbt-ki',
    'im2row-scan-ab-ik',
    'im2row-scan-abt-ik',
    'im2row-scan-abt-ki',
    'im2row-scan-atbt-ki',
    'kn2col',
    'kn2col-as',
    'kn2row',
    'kn2row-aa-ab',
    'kn2row-aa-abt',
    'kn2row-aa-atb',
    'kn2row-aa-atbt',
    'kn2row-as',
    'winograd-2-3',
    'winograd-2-3-vec-4',
    'winograd-2x2-3x3',
    'winograd-2x2-3x3-vec-16',
    'winograd-2x2-3x3-vec-4',
    'winograd-2x2-3x3-vec-8',
    'winograd-3-3',
    'winograd-3-3-vec-4',
    'winograd-3x3-3x3',
    'winograd-3x3-3x3-vec-16',
    'winograd-3x3-3x3-vec-4',
    'winograd-3x3-3x3-vec-8',
    'winograd-4x4-3x3',
    'winograd-4x4-3x3-vec-16',
    'winograd-4x4-3x3-vec-4',
    'winograd-4x4-3x3-vec-8',
    'winograd-2-5',
    'winograd-2-5-vec-4',
    'winograd-2x2-5x5',
    'winograd-2x2-5x5-vec-16',
    'winograd-2x2-5x5-vec-4',
    'winograd-2x2-5x5-vec-8',
    'winograd-3-5',
    'winograd-3-5-vec-4',
    'winograd-3x3-5x5',
    'winograd-3x3-5x5-vec-16',
    'winograd-3x3-5x5-vec-4',
    'winograd-3x3-5x5-vec-8',
    'winograd-4x4-5x5',
    'winograd-4x4-5x5-vec-16',
    'winograd-4x4-5x5-vec-4',
    'winograd-4x4-5x5-vec-8',
    'conv-1x1-gemm-ab-ik',
    'conv-1x1-gemm-ab-ki',
    'conv-1x1-gemm-abt-ik',
    'conv-1x1-gemm-abt-ki',
    'conv-1x1-gemm-atb-ik',
    'conv-1x1-gemm-atb-ki',
    'conv-1x1-gemm-atbt-ik',
    'conv-1x1-gemm-atbt-ki',
    'mec-col',
    'mec-row-partition'
])

DIRECT = list(range(0, 1))
IM2 = list(range(1, 21))
KN2 = list(range(21, 29))
WINO3 = list(range(29, 45))
WINO5 = list(range(45, 61))
CONV1 = list(range(61, 69))
MEC = list(range(69, 71))


def df_for_prim(df, prim, transform=False):
    if transform:
        return df[["c", "im"] + [prim]]
    else:
        return df[list(VARIABLES) + [prim]]


def settings_for_var(df, var):
    assert(var in VARIABLES)
    return list(map(int, list(set(df[var]))))