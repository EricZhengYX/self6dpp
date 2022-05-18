_base_ = ["../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/lm_pbr/my_config/ape"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=False,
    CHANGE_BG_PROB=0.5,  # 0.5
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="iaa_custom",
    COLOR_AUG_BG_REPLACE="datasets/VOCdevkit/VOC2012/JPEGImages",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.4, GaussianBlur((0., 3.))),"
        "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
        "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
        "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
        "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
        "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
        "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"  # maybe remove for det
        "], random_order=True)"
        # cosy+aae
    ),
    POSE_VARIATED_AUG=dict(
        OVERALL_PROB=0.3,
        CROP=dict(
            PERCENT=0.1
        ),
        ROT=dict(
            MAX_DEGREE=360,
        ),
        TRANS=dict(
            UPPER_LIM=0.1,
            LOWER_LIM=-0.1,
        ),
        ZOOM=dict(
            UPPER_LIM=1.25,
            LOWER_LIM=0.75,
        )
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=24,  # 24
    TOTAL_EPOCHS=100,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    CLIP_GRAD=100,
)

DATASETS = dict(
    TRAIN=("lm_pbr_ape_train",),  # TRAIN=("lm_pbr_ape_train",), lm_real_ape_train, lmo_test, lm_real_ape_test
    TEST=("lmo_test", "lm_real_ape_test",),
    DET_FILES_TEST=(
        "datasets/BOP_DATASETS/lmo/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lmo_pbr_lmo_test_16e.json",
        "datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_test_16e.json",
    ),
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    POSE_NET=dict(
        NAME="GDRN_double_mask_double_vf",  # GDRN_double_mask_double_vf, GDRN_double_mask
        XYZ_ONLINE=False,
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/resnest50d",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(4,),
            ),
        ),
        ## geo head: Mask, XYZ, Region, VF
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="TopDownDoubleMaskDoubleVFXyzRegionHead",  # TopDownDoubleMaskXyzRegionHead, TopDownDoubleMaskDoubleVFXyzRegionHead
                in_dim=2048,  # this is num out channels of backbone conv feature
            ),
            NUM_REGIONS=64,
            VF_CLASS_AWARE=False,
            NUM_CHANNAL_VF=16,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(
                norm="GN",
                act="gelu",
                type="ConvPnPNetAll",
            ),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            MASK_ATTENTION="concat",
            WITH_VF="both",
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            # xyz loss ----------------------------
            XYZ_LOSS_TYPE="L1",  # L1 | CE_coor
            XYZ_LOSS_MASK_GT="visib",  # trunc | visib | obj
            XYZ_LW=1.0,
            # mask loss ---------------------------
            MASK_LOSS_TYPE="BCE",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            # full mask loss ---------------------------
            FULL_MASK_LOSS_TYPE="BCE",  # L1 | BCE | CE
            FULL_MASK_LW=1.0,
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=0.05,
            # vf loss ---------------------------
            VF_LOSS_TYPE="L1+Cos",
            VIS_VF_LW=1.0,
            FULL_VF_LW=1.0,
            # vf-rt loss ---------------------------
            VF_RT_LOSS_TYPE="L1+Cos",
            VF_RT_LW=1.0,
            # pm loss --------------
            PM_LOSS_SYM=True,  # NOTE: sym loss
            PM_R_ONLY=True,  # only do R loss in PM
            PM_LW=10.0,
            # centroid loss -------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            # z loss -----------
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
        ),
    ),
)

TEST = dict(
    EVAL_PERIOD=5, # count in epochs
    VIS=False,
    TEST_BBOX_TYPE="est"
)  # gt | est
