import os
import os.path as osp
from ref.lmo_full import objects
import shutil

ignore = []
DELETE_OLD = True


def build_content(file_name, cls_name, with_depth=False, with_weak=False):
    content = \
'''_base_ = ["../ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config_depth_weak/lmo/{}"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_{}_train",),
    TEST=("lmo_{}_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_{}_pbr.pth",
    POSE_NET=dict(
        SELF_LOSS_CFG=dict(
            GEOM_LW={},
        )
    )
)
REPJ_REFINE = dict(
    ENABLE={},
)
'''.format(file_name, cls_name, cls_name, cls_name, 100 if with_depth else 0, 'True' if with_weak else 'False')
    return content


if __name__ == "__main__":
    assert osp.exists("ssLMO_base.py"), "Working dir should looks like \'configs/self6dpp/new_cfg_depth_weak/ssLMO\'"
    for obj_name in objects:
        if obj_name not in ignore:
            if osp.exists(obj_name) and DELETE_OLD:
                shutil.rmtree(obj_name)
            elif osp.exists(obj_name) and not DELETE_OLD:
                continue
            os.makedirs(obj_name, exist_ok=True)
            # 1
            file_name = "{}_oo.py".format(obj_name)
            with open(osp.join(obj_name, file_name), 'w') as f:
                f.write(build_content(file_name, obj_name, with_depth=False, with_weak=False))
            # 2
            file_name = "{}_ow.py".format(obj_name)
            with open(osp.join(obj_name, file_name), 'w') as f:
                f.write(build_content(file_name, obj_name, with_depth=False, with_weak=True))
            # 3
            file_name = "{}_wo.py".format(obj_name)
            with open(osp.join(obj_name, file_name), 'w') as f:
                f.write(build_content(file_name, obj_name, with_depth=True, with_weak=False))
            # 4
            file_name = "{}_ww.py".format(obj_name)
            with open(osp.join(obj_name, file_name), 'w') as f:
                f.write(build_content(file_name, obj_name, with_depth=True, with_weak=True))
