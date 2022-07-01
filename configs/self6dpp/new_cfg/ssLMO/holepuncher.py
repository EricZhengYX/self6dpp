_base_ = ["ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lmo/holepuncher"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_holepuncher_train",),
    TEST=("lmo_holepuncher_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_holepuncher_pbr.pth",
)
