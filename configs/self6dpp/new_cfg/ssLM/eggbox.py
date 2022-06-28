_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/base/eggbox"

DATASETS = dict(
    TRAIN=("lm_real_eggbox_train",),
    TEST=("lm_real_eggbox_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_eggbox_pbr.pth",
)
