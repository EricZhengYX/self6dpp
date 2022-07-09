_base_ = ["base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/benchvise"

DATASETS = dict(
    TRAIN=("lm_pbr_benchvise_train",),
    TEST=("lmo_benchvise_bop_test", "lm_real_benchvise_test"),
)
