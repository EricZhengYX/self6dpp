_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/can"

DATASETS = dict(
    TRAIN=("lm_pbr_can_train",),
    TEST=("lmo_can_bop_test", "lm_real_can_test"),
)
