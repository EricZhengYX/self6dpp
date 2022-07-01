_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/driller"

DATASETS = dict(
    TRAIN=("lm_real_driller_train",),
    TEST=("lm_real_driller_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_driller_pbr.pth",
)
