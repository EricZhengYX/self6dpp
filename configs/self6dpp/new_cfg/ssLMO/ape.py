_base_ = ["ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lmo/base/ape"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_ape_train",),
    TEST=("lmo_ape_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_ape_pbr.pth",
)
