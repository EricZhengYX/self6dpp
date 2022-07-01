_base_ = ["ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lmo/glue"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_glue_train",),
    TEST=("lmo_glue_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_glue_pbr.pth",
)
