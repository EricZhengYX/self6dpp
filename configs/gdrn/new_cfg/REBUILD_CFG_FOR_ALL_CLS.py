import os
import os.path as osp
from ref.lm_full import objects as lm_objs
from ref.lmo_full import objects as lmo_objs

ignore = ["bowl", "cup"]
DELETE_OLD = True


def build_content(cls_name, only_LM=True):
    if only_LM:
        content = \
'''_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/{}"

DATASETS = dict(
    TRAIN=("lm_pbr_{}_train",),
    TEST=("lm_real_{}_test",),
    DET_FILES_TEST=(
        "datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_test_16e.json",
    ),
)
'''.format(cls_name, cls_name, cls_name, cls_name)
    else:
        content = \
'''_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/{}"

DATASETS = dict(
    TRAIN=("lm_pbr_{}_train",),
    TEST=("lmo_{}_bop_test", "lm_real_{}_test"),
)
'''.format(cls_name, cls_name, cls_name, cls_name)
    return content


if __name__ == "__main__":
    assert osp.exists("LM_base.py"), "Working dir should looks like \'configs/gdrn/new_cfg/ssLM\'"
    for obj_name in lm_objs:
        if obj_name not in ignore:
            file_name = "{}.py".format(obj_name)
            if osp.exists(file_name) and DELETE_OLD:
                os.remove(file_name)
            elif osp.exists(file_name) and not DELETE_OLD:
                continue

            with open(file_name, 'w') as f:
                f.write(build_content(obj_name, only_LM=obj_name not in lmo_objs))
