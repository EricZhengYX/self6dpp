_base_ = ["../ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config_depth_weak/lm/lamp_oo.py"

DATASETS = dict(
    TRAIN=("lm_real_lamp_train",),
    TEST=("lm_real_lamp_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_lamp_pbr.pth",
    POSE_NET=dict(
        SELF_LOSS_CFG=dict(
            GEOM_LW=0,
        )
    )
)
REPJ_REFINE = dict(
    ENABLE=False,
)
