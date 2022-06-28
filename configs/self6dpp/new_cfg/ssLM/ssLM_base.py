_base_ = ["../base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/base"

SOLVER = dict(
    TOTAL_EPOCHS=100,
    CHECKPOINT_PERIOD=30,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-6, weight_decay=0),
)

DATASETS = dict(
    TRAIN=("YOU SHOULD REPLACE THIS",),  # real data, eg. lm_real_ape_train
    TRAIN2_RATIO=0.0,
    TEST=("YOU SHOULD REPLACE THIS",),  # eg. lm_real_ape_test
    # for self-supervised training
    DET_FILES_TRAIN=(
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_withYolov4PbrBbox_wDeepimPbrPose_lm_13_train.json",
    ),
    DET_THR_TRAIN=0.5,
    DET_FILES_TEST=(
        "datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_test_16e.json",
    ),
)

TEST = dict(
    EVAL_PERIOD=20,  # count in epochs
)
