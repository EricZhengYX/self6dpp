import pickle
import warnings
import mmcv
from adding.folder_checker import validate_lm
from adding.DIB_extended.dr_utils.dr_utils import load_plys
import json
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch


BOP_ROOT_PTH = osp.abspath(osp.curdir) + "/datasets/BOP_DATASETS/"
DS_ROOT_PTH = (
    BOP_ROOT_PTH + "{}/test"
)  # where to find depth/mask/mask_visib/rgb... and so on
MODEL_ROOT_PTH = BOP_ROOT_PTH + "{}/models"  # where to find models like obj_000000.ply
SEQ_DICT = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
    "ape": 1,
    "benchvise": 2,
    "bowl": 3,
    "camera": 4,
    "can": 5,
    "cat": 6,
    "cup": 7,
    "driller": 8,
    "duck": 9,
    "eggbox": 10,
    "glue": 11,
    "holepuncher": 12,
    "iron": 13,
    "lamp": 14,
    "phone": 15,
}


class DatasetInfo:
    def __init__(
        self,
        dataset_name="lm",
        seq=range(1, 16),
        noise_level=0,
        load_extr=True,
        device="cuda",
    ):
        self.dn = dataset_name
        self.seq = list(map(lambda x: int(x), seq))
        assert all([0 < s < 16 for s in self.seq])
        ds_root_pth = DS_ROOT_PTH.format(dataset_name)
        model_root_pth = MODEL_ROOT_PTH.format(dataset_name)
        if dataset_name == "lm":
            validate_lm(
                ds_root_pth=ds_root_pth, model_root_pth=model_root_pth, seq=self.seq
            )
        else:
            raise NotImplementedError
        camera_json = osp.abspath(osp.join(ds_root_pth, "..", "camera.json"))
        assert osp.exists(camera_json), camera_json
        self.K, self.H, self.W = self.__cam_setting(camera_json)

        for i in self.seq:
            seq_name = SEQ_DICT[i]
            setattr(
                self,
                seq_name,
                SeqInfo(
                    osp.join(ds_root_pth, str(i).zfill(6)),
                    model_root_pth,
                    i,
                    noise_level,
                    load_extr,
                    device,
                ),
            )

    def __cam_setting(self, json_pth: str):
        d = mmcv.load(json_pth)
        K = [
            [d["fx"], 0,       d["cx"]],
            [0,       d["fy"], d["cy"]],
            [0,       0,             1],
        ]
        h, w = d["height"], d["width"]
        return np.array(K), h, w


class SeqInfo:
    def __init__(
        self,
        ds_pth: str,
        model_pth: str,
        model_idx: int,
        noise_level,
        load_extr=True,
        device="cuda",
    ):
        # Path
        self.root_pth = ds_pth
        self.model_idx = model_idx
        self.model_pth = osp.join(
            model_pth, "obj_{}.ply".format(str(model_idx).zfill(6))
        )
        self.__models_info_pth = osp.join(model_pth, "models_info.json")
        self.rgb_pth = osp.join(ds_pth, "rgb")
        self.depth_pth = osp.join(ds_pth, "depth")
        self.mask_pth = osp.join(ds_pth, "mask")
        self.mask_visib_pth = osp.join(ds_pth, "mask_visib")
        cam_para_gt_format = "pkl"
        cam_para_gt_filename = (
            "cam_para_gt.{}".format(cam_para_gt_format)
            if noise_level == 0
            else "cam_para_gt_{n}.{s}".format(n=noise_level, s=cam_para_gt_format)
        )
        self.cam_para_gt_pth = osp.join(ds_pth, cam_para_gt_filename)
        self.pose_inf_pth = osp.join(ds_pth, "pose_inf.json")
        self.scene_gt_pth = osp.join(ds_pth, "scene_gt.json")
        self.scene_gt_info_pth = osp.join(ds_pth, "scene_gt_info.json")

        # device
        self.device = device

        # load jsons/pkl
        with open(self.__models_info_pth, "r") as json_file:
            __models_info = json.load(json_file)
        self.__this_model_info_ori = __models_info[str(model_idx)]
        self.__this_model_info = {}
        for k in {"diameter", "min_x", "min_y", "min_z", "size_x", "size_y", "size_z"}:
            self.__this_model_info[k] = self.__this_model_info_ori[k] / 1000

        # Maybe raise a FileNotFoundErr
        if load_extr:
            if cam_para_gt_format == "pkl":
                with open(self.cam_para_gt_pth, "br") as f:
                    self.__cam_para_gt = pickle.load(f)
            elif cam_para_gt_format == "json":
                with open(self.cam_para_gt_pth, "r") as f:
                    self.__cam_para_gt = json.load(f)
            else:
                warnings.warn(
                    "Unknown  file format: {}, cam_para_gt is not loaded".format(
                        self.cam_para_gt_pth
                    )
                )
                self.__cam_para_gt = None
        else:
            self.__cam_para_gt = None

        """
        try:
            with open(self.pose_inf_pth, 'r') as json_file:
                self.__pose_inf = json.load(json_file)
        except:
            warnings.warn('Missing pose_inf.json in seq{}'.format(model_idx))
            self.__pose_inf = None
        """
        with open(self.scene_gt_pth, "r") as json_file:
            self.__scene_gt = json.load(json_file)
        with open(self.scene_gt_info_pth, "r") as json_file:
            self.__scene_gt_info = json.load(json_file)

        # model cache
        self.model = None
        # Count items
        _cnt = 0
        for f in os.listdir(self.rgb_pth):
            if osp.isfile(osp.join(self.rgb_pth, f)):
                _cnt += 1
        self.__len = _cnt

    def __len__(self):
        return self.__len

    def __repr__(self):
        return "'{n}' at: {r}".format(n=SEQ_DICT[self.model_idx], r=self.root_pth)

    def get_one_img_dir(self, idx: int):
        return osp.join(self.rgb_pth, "{}.png".format(str(idx).zfill(6)))

    def get_one_img(self, idx: int):
        img_pth = self.get_one_img_dir(idx)
        assert osp.exists(img_pth)
        return np.array(Image.open(img_pth))

    def get_2dbbox(self, idx: int):
        """
        Return 2d bbox like: [x min. y min, x max, y max]
        """
        mask = self.get_one_mask(idx)

        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]

        if len(horizontal_indicies) < 2 or len(vertical_indicies) < 2:
            bbox = np.array([0, 0, 0, 0])
        else:
            bbox = np.array(
                [
                    horizontal_indicies[0],
                    vertical_indicies[0],
                    horizontal_indicies[-1],
                    vertical_indicies[-1],
                ]
            )

        return bbox

    def get_one_mask_dir(self, idx: int):
        return osp.join(
            self.mask_pth, "{i}_{f}.png".format(i=str(idx).zfill(6), f="000000")
        )

    def get_one_mask(self, idx: int, shrink_factor=1):
        img_pth = self.get_one_mask_dir(idx)
        assert osp.exists(img_pth)
        return (np.array(Image.open(img_pth)) != 0)[::shrink_factor, ::shrink_factor]

    def get_one_depth_dir(self, idx: int):
        return osp.join(self.depth_pth, "{}.png".format(str(idx).zfill(6)))

    def get_one_depth(self, idx: int):
        img_pth = self.get_one_depth_dir(idx)
        assert osp.exists(img_pth)
        return np.array(Image.open(img_pth))

    def get_one_mask_visib_dir(self, idx: int):
        return osp.join(self.mask_visib_pth, "{}.png".format(str(idx).zfill(6)))

    def get_one_mask_visib(self, idx: int):
        img_pth = self.get_one_mask_visib_dir(idx)
        assert osp.exists(img_pth)
        return np.array(Image.open(img_pth)) != 0

    def get_cam_para(self, s_i: int, t_i: int):
        if self.__cam_para_gt is None:
            warnings.warn("No cam_para_gt loaded!")
            return None
        if s_i == t_i:
            return torch.eye(4)
        if s_i < t_i:
            r = torch.reshape(
                torch.tensor(self.__cam_para_gt[str(s_i)][str(t_i)]["rot"]), (3, 3)
            )
            t = torch.reshape(
                torch.tensor(self.__cam_para_gt[str(s_i)][str(t_i)]["tran"]), (3, 1)
            )
            return torch.vstack((torch.hstack((r, t)), torch.tensor([0, 0, 0, 1])))
        else:
            r = torch.reshape(
                torch.tensor(self.__cam_para_gt[str(t_i)][str(s_i)]["rot"]), (3, 3)
            )
            t = torch.reshape(
                torch.tensor(self.__cam_para_gt[str(t_i)][str(s_i)]["tran"]), (3, 1)
            )
            _t = torch.vstack((torch.hstack((r, t)), torch.tensor([0, 0, 0, 1])))
            return torch.inverse(_t)

    def get_pose_inf(self, idx: int):
        if self.__pose_inf is None:
            warnings.warn("No pose_inf.json loaded!")
            return None
        pose_dict = self.__pose_inf[str(idx)]
        r = torch.reshape(torch.tensor(pose_dict["rot"]), (3, 3))
        t = torch.reshape(torch.tensor(pose_dict["tran"]), (3, 1))
        return torch.vstack((torch.hstack((r, t)), torch.tensor([0, 0, 0, 1])))

    def get_pose_gt(self, idx: int):
        pose_dict = self.__scene_gt[str(idx)][0]
        r = torch.reshape(torch.tensor(pose_dict["cam_R_m2c"]), (3, 3))
        t = torch.reshape(torch.tensor(pose_dict["cam_t_m2c"]), (3, 1)) / 1000
        return torch.vstack((torch.hstack((r, t)), torch.tensor([0, 0, 0, 1])))

    def get_roi(self, idx: int, mode="xxyy"):
        roi_dict = self.__scene_gt_info[str(idx)][0]
        roi = list(map(int, roi_dict["bbox_obj"]))
        if mode.lower() == "xycw":
            return roi
        elif mode.lower() == "xxyy":
            return [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]]
        else:
            raise NotImplementedError

    def get_roi_visib(self, idx: int, mode="xxyy"):
        roi_dict = self.__scene_gt_info[str(idx)][0]
        roi = list(map(int, roi_dict["bbox_visib"]))
        if mode == "xycw":
            return roi
        elif mode == "xxyy":
            return [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]]
        else:
            raise NotImplementedError

    def get_model(self):
        if self.model is None:
            model = load_plys(self.model_pth, device=self.device)
            if isinstance(model, list):
                assert len(model) == 1, "Trying to load multiply models"
                self.model = model[0]
            else:
                self.model = model
        return self.model

    def get_model_info(self):
        return self.__this_model_info


if __name__ == "__main__":
    import matplotlib
    from matplotlib import pyplot as plt

    matplotlib.use("TkAgg")

    ds_info = DatasetInfo(seq=[6], noise_level=10, device="cpu")
    cat_seq: SeqInfo = ds_info.cat

    plt.imshow(cat_seq.get_one_mask(0))
    plt.show()
    pass
