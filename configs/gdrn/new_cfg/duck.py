_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/duck"

DATASETS = dict(
    TRAIN=("lm_pbr_duck_train",),
    TEST=("lmo_duck_bop_test", "lm_real_duck_test"),
)
