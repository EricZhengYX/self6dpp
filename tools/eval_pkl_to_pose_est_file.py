import mmcv
import re
import numpy as np
from core.self6dpp.datasets.lm_dataset_d2 import LM_DICT

file_name = "/home/eric/tb_remote/self-my-config-test_lmo_NoBopTest_ape_train_preds.pkl"
record = mmcv.load(file_name)
save_name_suffix = "lmo_NoBopTest_ape_train_e43474"

result = {}
for obj_name, single_obj_dict in record.items():
    obj_id = LM_DICT[obj_name]
    for file_pth, single_img_dict in single_obj_dict.items():
        pth_split = file_pth.split('/')
        seq_int = int(pth_split[-3])
        img_int = int(re.search("\d+", pth_split[-1]).group())
        key = "{}/{}".format(seq_int, img_int)
        r = single_img_dict["R"]
        t = single_img_dict["t"]

        est_dict = {
            "pose_est": np.hstack((r, t[:, None])).tolist(),
            "obj_id": obj_id,
        }
        if key not in result:
            result[key] = [est_dict]
        else:
            result[key].append(est_dict)
mmcv.dump(result, "/home/eric/tb_remote/pose_est_{}.json".format(save_name_suffix))
