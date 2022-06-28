_base_ = ["ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lmo/base/cat"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_cat_train",),
    TEST=("lmo_cat_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_cat_pbr.pth",
)
