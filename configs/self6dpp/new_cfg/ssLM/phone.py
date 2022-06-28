_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/base/phone"

DATASETS = dict(
    TRAIN=("lm_real_phone_train",),
    TEST=("lm_real_phone_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_phone_pbr.pth",
)
