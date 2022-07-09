_base_ = ["../base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lmo/base"

SOLVER = dict(
    TOTAL_EPOCHS=40,
    WARMUP_ITERS=1000,
    CHECKPOINT_PERIOD=10,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-6, weight_decay=0),
)

DATASETS = dict(
    TRAIN=("YOU SHOULD REPLACE THIS",),  # real data, eg. lmo_NoBopTest_ape_train
    TRAIN2_RATIO=0.0,
    TEST=("YOU SHOULD REPLACE THIS",),  # eg. lmo_ape_bop_test
    # for self-supervised training
    DET_FILES_TRAIN=(
        "datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_withYolov4PbrBbox_wDeepimPbrPose_lmo_NoBopTest_train.json",
    ),
    DET_THR_TRAIN=0.5,
    DET_FILES_TEST=(
        "datasets/BOP_DATASETS/lmo/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lmo_pbr_lmo_test_16e.json",
    ),
)

TEST = dict(
    EVAL_PERIOD=1,  # count in epochs
)
