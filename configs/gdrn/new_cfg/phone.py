_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/phone"

DATASETS = dict(
    TRAIN=("lm_pbr_phone_train",),
    TEST=("lmo_phone_bop_test", "lm_real_phone_test"),
)
