_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/holepuncher"

DATASETS = dict(
    TRAIN=("lm_pbr_holepuncher_train",),
    TEST=("lmo_holepuncher_bop_test", "lm_real_holepuncher_test"),
)
