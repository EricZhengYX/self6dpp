_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/can"

DATASETS = dict(
    TRAIN=("lm_real_can_train",),
    TEST=("lm_real_can_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_can_pbr.pth",
)
