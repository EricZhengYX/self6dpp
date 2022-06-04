import numpy as np


def load_transform_dat(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()
        assert lines[0].split() == ['12']
        mtx = np.array([float(l.split()[1]) for l in lines[1:]])
        mtx.resize(3, 4)
    return mtx[:, :3], mtx[:, 3, None]
