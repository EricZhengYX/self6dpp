_base_ = ["base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/driller"

DATASETS = dict(
    TRAIN=("lm_pbr_driller_train",),
    TEST=("lmo_driller_bop_test", "lm_real_driller_test"),
)
