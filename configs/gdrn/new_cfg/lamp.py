_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/lamp"

DATASETS = dict(
    TRAIN=("lm_pbr_lamp_train",),
    TEST=("lmo_lamp_bop_test", "lm_real_lamp_test"),
)
