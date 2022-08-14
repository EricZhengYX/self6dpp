_base_ = ["../ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config_depth_weak/lm/phone_wo.py"

DATASETS = dict(
    TRAIN=("lm_real_phone_train",),
    TEST=("lm_real_phone_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_phone_pbr.pth",
    POSE_NET=dict(
        SELF_LOSS_CFG=dict(
            GEOM_LW=100,
        )
    )
)
REPJ_REFINE = dict(
    ENABLE=False,
)
