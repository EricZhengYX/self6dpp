_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/lamp"

DATASETS = dict(
    TRAIN=("lm_real_lamp_train",),
    TEST=("lm_real_lamp_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_lamp_pbr.pth",
)
