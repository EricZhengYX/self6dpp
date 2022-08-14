_base_ = ["../ssLMO_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config_depth_weak/lmo/duck_ww.py"

DATASETS = dict(
    TRAIN=("lmo_NoBopTest_duck_train",),
    TEST=("lmo_duck_bop_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_duck_pbr.pth",
    POSE_NET=dict(
        SELF_LOSS_CFG=dict(
            GEOM_LW=100,
        )
    )
)
REPJ_REFINE = dict(
    ENABLE=True,
)
