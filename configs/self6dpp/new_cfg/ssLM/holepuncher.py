_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/base/holepuncher"

DATASETS = dict(
    TRAIN=("lm_real_holepuncher_train",),
    TEST=("lm_real_holepuncher_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_holepuncher_pbr.pth",
)
