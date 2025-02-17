_base_ = ["ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lmo/can"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_can_train",),
    TEST=("lmo_can_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_can_pbr.pth",
)
