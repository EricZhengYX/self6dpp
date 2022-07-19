_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/cat"

DATASETS = dict(
    TRAIN=("lm_pbr_cat_train",),
    TEST=("lmo_cat_bop_test", "lm_real_cat_test"),
)
