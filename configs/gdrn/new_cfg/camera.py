_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/camera"

DATASETS = dict(
    TRAIN=("lm_pbr_camera_train",),
    TEST=("lmo_camera_bop_test", "lm_real_camera_test"),
)
