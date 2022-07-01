_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/benchvise"

DATASETS = dict(
    TRAIN=("lm_real_benchvise_train",),
    TEST=("lm_real_benchvise_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_benchvise_pbr.pth",
)
