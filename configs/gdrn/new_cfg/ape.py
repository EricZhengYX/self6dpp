_base_ = ["base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/ape"

DATASETS = dict(
    TRAIN=("lm_pbr_ape_train",),
    TEST=("lmo_ape_bop_test", "lm_real_ape_test"),
)
