_base_ = ["base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/iron"

DATASETS = dict(
    TRAIN=("lm_pbr_iron_train",),
    TEST=("lmo_iron_bop_test", "lm_real_iron_test"),
)
