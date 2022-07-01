import os
import os.path as osp
from ref.lm_full import objects

ignore = ["bowl", "cup"]
DELETE_OLD = True


def build_content(cls_name):
    content = \
'''_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/{}"

DATASETS = dict(
    TRAIN=("lm_real_{}_train",),
    TEST=("lm_real_{}_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_{}_pbr.pth",
)
'''.format(cls_name, cls_name, cls_name, cls_name)
    return content


if __name__ == "__main__":
    assert osp.exists("ssLM_base.py"), "Working dir should looks like \'configs/self6dpp/new_cfg/ssLM\'"
    for obj_name in objects:
        if obj_name not in ignore:
            file_name = "{}.py".format(obj_name)
            if osp.exists(file_name) and DELETE_OLD:
                os.remove(file_name)
            elif osp.exists(file_name) and not DELETE_OLD:
                continue

            with open(file_name, 'w') as f:
                f.write(build_content(obj_name))
