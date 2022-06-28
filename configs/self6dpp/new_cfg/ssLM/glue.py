_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/base/glue"

DATASETS = dict(
    TRAIN=("lm_real_glue_train",),
    TEST=("lm_real_glue_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_glue_pbr.pth",
)
