_base_ = ["ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config/lm/camera"

DATASETS = dict(
    TRAIN=("lm_real_camera_train",),
    TEST=("lm_real_camera_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_camera_pbr.pth",
)
