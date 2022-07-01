import os
import os.path as osp
from ref.lmo_full import objects

ignore = []
DELETE_OLD = True


def build_content(cls_name):
    content = \
'''_base_ = ["ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lmo/{}"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_{}_train",),
    TEST=("lmo_{}_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_{}_pbr.pth",
)
'''.format(cls_name, cls_name, cls_name, cls_name)
    return content


if __name__ == "__main__":
    assert osp.exists("ssLMO_base.py"), "Working dir should looks like \'configs/self6dpp/new_cfg/ssLMO\'"
    for obj_name in objects:
        if obj_name not in ignore:
            file_name = "{}.py".format(obj_name)
            if osp.exists(file_name) and DELETE_OLD:
                os.remove(file_name)
            elif osp.exists(file_name) and not DELETE_OLD:
                continue

            with open(file_name, 'w') as f:
                f.write(build_content(obj_name))
