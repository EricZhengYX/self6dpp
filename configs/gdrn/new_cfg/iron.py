_base_ = ["LM_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/new_cfg/iron"

DATASETS = dict(
    TRAIN=("lm_pbr_iron_train",),
    TEST=("lm_real_iron_test",),
    DET_FILES_TEST=(
        "datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_test_16e.json",
    ),
)
