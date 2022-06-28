_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/base/ape"

DATASETS = dict(
    TRAIN=("lm_real_ape_train",),
    TEST=("lm_real_ape_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_ape_pbr.pth",
)
