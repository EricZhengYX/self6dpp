_base_ = ["../ssLM_base.py"]

OUTPUT_DIR = "output/self6dpp/new_config_depth_weak/lm/iron_ww.py"

DATASETS = dict(
    TRAIN=("lm_real_iron_train",),
    TEST=("lm_real_iron_test",),
)
MODEL = dict(
    # synthetically pretrained model
    WEIGHTS="datasets/GS6D/LMLMOpbr/model_iron_pbr.pth",
    POSE_NET=dict(
        SELF_LOSS_CFG=dict(
            GEOM_LW=100,
        )
    )
)
REPJ_REFINE = dict(
    ENABLE=True,
)
