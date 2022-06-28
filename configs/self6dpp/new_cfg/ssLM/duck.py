_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/base/duck"

DATASETS = dict(
    TRAIN=("lm_real_duck_train",),
    TEST=("lm_real_duck_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_duck_pbr.pth",
)
