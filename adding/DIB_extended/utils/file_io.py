import warnings
import numpy as np
import os
from PIL import Image


class LM_loader:
    def __init__(self, load_path):
        if not isinstance(load_path, str):
            raise NotImplementedError
        self.pth = load_path
        self.__path_check()

        self.__im = None
        self.__D = None
        self.__R = None
        self.__T = None

    def __path_check(self):
        _exist = os.path.exists(self.pth)
        if not _exist:
            warnings.warn('Linemod Loader: Path not exist!')
        return _exist

    # def set_path(self, path):
    #     if not isinstance(load_path, str):
    #         raise NotImplementedError
    #     self.pth = load_path
    #     self.__path_check()

    def load(self, model_id: int, D=True, R=True, T=True, im=True):
        self.__load_everything(model_id, D, R, T, im)
        return self.__im, self.__D, self.__R, self.__T

    def __load_everything(self, model_id: int, D, R, T, im):

        if not D:
            warnings.warn('Linemod Loader: Depth is not required to be loaded!')
        else:
            d_pth = self.pth + 'depth{}.dpt'.format(model_id)
            if not os.path.exists(d_pth):
                warnings.warn('Linemod Loader: Depth file not exist!')
            else:
                self.__load_d(d_pth)

        if not R:
            warnings.warn('Linemod Loader: Rotation is not required to be loaded!')
        else:
            r_pth = self.pth + 'rot{}.rot'.format(model_id)
            if not os.path.exists(r_pth):
                warnings.warn('Linemod Loader: Rotation file not exist!')
            else:
                self.__load_r(r_pth)

        if not T:
            warnings.warn('Linemod Loader: Translation is not required to be loaded!')
        else:
            t_pth = self.pth + 'tra{}.tra'.format(model_id)
            if not os.path.exists(t_pth):
                warnings.warn('Linemod Loader: Translation file not exist!')
            else:
                self.__load_t(t_pth)

        if not im:
            warnings.warn('Linemod Loader: Image is not required to be loaded!')
        else:
            im_pth = self.pth + 'color{}.jpg'.format(model_id)
            if not os.path.exists(im_pth):
                warnings.warn('Linemod Loader: Image file not exist!')
            else:
                self.__load_im(im_pth)

    def __load_r(self, pth: str):
        with open(pth, 'r') as f:
            lines = f.readlines()
            assert lines[0].split() == ['3', '3']
            self.__R = np.array([list(map(float, l.split())) for l in lines[1:]])

    def __load_im(self, pth: str):
        img = Image.open(pth)
        self.__im = img

    def __load_t(self, pth: str):
        with open(pth, 'r') as f:
            lines = f.readlines()
            assert lines[0].split() == ['1', '3']
            self.__T = np.array([list(map(float, lines[1:]))])
            self.__T.resize(3, 1)  # 需要添加此行，保证形状为3*1而不是1*3

    # Borrow from: https://github.com/paroj/linemod_dataset/blob/master/read.py
    def __load_d(self, pth: str):
        """
        read a depth image

        @return uint16 image of distance in [mm]"""
        dpt = open(pth, "rb")
        rows = np.frombuffer(dpt.read(4), dtype=np.int32)[0]
        cols = np.frombuffer(dpt.read(4), dtype=np.int32)[0]

        self.__D = np.fromfile(dpt, dtype=np.uint16).reshape((rows, cols))
