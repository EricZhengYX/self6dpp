_base_ = ["base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/eggbox"

DATASETS = dict(
    TRAIN=("lm_pbr_eggbox_train",),
    TEST=("lmo_eggbox_bop_test", "lm_real_eggbox_test"),
)
