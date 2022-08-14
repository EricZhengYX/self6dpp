_base_ = ["../ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config_depth_weak/lmo/can_ow.py"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_can_train",),
    TEST=("lmo_can_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_can_pbr.pth",
    POSE_NET=dict(
        SELF_LOSS_CFG=dict(
            GEOM_LW=0,
        )
    )
)
REPJ_REFINE = dict(
    ENABLE=True,
)
