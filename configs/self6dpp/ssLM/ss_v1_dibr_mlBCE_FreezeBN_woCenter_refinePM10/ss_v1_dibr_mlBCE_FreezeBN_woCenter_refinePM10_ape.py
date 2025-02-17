_base_ = ["../../../_base_/self6dpp_base.py"]  # ["../../../../_base_/self6dpp_base.py"]

OUTPUT_DIR = "output/self6dpp/ssLM/ss_v1_dibr_mlBCE_FreezeBN_woCenter_refinePM10/ape"
INPUT = dict(
    WITH_DEPTH=True,
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=False,
    CHANGE_BG_PROB=0.5,
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
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
)

SOLVER = dict(
    IMS_PER_BATCH=6,  # maybe need to be < 24
    TOTAL_EPOCHS=100,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=100,  # NOTE: only real data, iterations are very small
    CLIP_GRADIENTS=dict(ENABLED=True, CLIP_TYPE="full_model", CLIP_VALUE=1.0),
)

DATASETS = dict(
    TRAIN=("lm_real_ape_train",),  # real data
    TRAIN2=("lm_pbr_ape_train",),  # synthetic data
    TRAIN2_RATIO=0.0,
    TEST=("lm_real_ape_test",),
    # for self-supervised training
    DET_FILES_TRAIN=(
        "datasets/BOP_DATASETS/lm/test/init_poses/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e_so_withYolov4PbrBbox_wDeepimPbrPose_lm_13_train.json",
    ),
    DET_THR_TRAIN=0.5,
    DET_FILES_TEST=(
        "datasets/BOP_DATASETS/lm/test/test_bboxes/yolov4x_640_test672_augCosyAAEGray_ranger_lm_pbr_lm_test_16e.json",
    ),
)

RENDERER = dict(DIFF_RENDERER="DIBR")  # DIBR | DIBR

MODEL = dict(
    # synthetically trained model
    WEIGHTS="output/gdrn/lm_pbr/resnest50d_a6_AugCosyAAEGray_BG05_mlBCE_lm_pbr_100e/ape/model_final_wo_optim-e8c99c96.pth",
    # init
    # ad10    rete5    te2
    # 50.86   98.57    91.81
    REFINER_WEIGHTS="",
    FREEZE_BN=True,
    SELF_TRAIN=True,  # whether to do self-supervised training
    WITH_REFINER=False,  # whether to use refiner
    LOAD_DETS_TRAIN=True,  # NOTE: load detections for self-train
    LOAD_DETS_TRAIN_WITH_POSE=True,  # NOTE: load pose_refine
    LOAD_DETS_TEST=True,
    EMA=dict(
        ENABLED=True,
        INIT_CFG=dict(decay=0.999, updates=0),  # epoch-based
        UPDATE_FREQ=10,  # update the mean teacher every n epochs
    ),
    POSE_NET=dict(
        NAME="GDRN",  # used module file name
        # NOTE: for self-supervised training phase, use offline labels should be more accurate
        XYZ_ONLINE=False,  # rendering xyz online
        XYZ_BP=True,  # calculate xyz from depth by backprojection
        NUM_CLASSES=13,
        USE_MTL=False,  # uncertainty multi-task weighting
        INPUT_RES=256,
        OUTPUT_RES=64,
        ## backbone
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
        ## geo head: Mask, XYZ, Region
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="TopDownMaskXyzRegionHead",
                in_dim=2048,  # this is num out channels of backbone conv feature
            ),
            NUM_REGIONS=64,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu"),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
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
            # region loss -------------------------
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=1.0,
            # pm loss --------------
            PM_LOSS_SYM=True,  # NOTE: sym loss
            PM_R_ONLY=True,  # only do R loss in PM
            PM_LW=1.0,
            # centroid loss -------
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            # z loss -----------
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
        ),
        SELF_LOSS_CFG=dict(
            # LAB space loss ------------------
            LAB_NO_L=True,
            LAB_LW=0.2,
            # MS-SSIM loss --------------------
            MS_SSIM_LW=1.0,
            # perceptual loss -----------------
            PERCEPT_LW=0.15,
            # mask loss (init, ren) -----------------------
            MASK_INIT_REN_LOSS_TYPE="RW_BCE",
            # MASK_INIT_REN_LOSS_TYPE="dice",
            MASK_INIT_REN_LW=1.0,
            # depth-based geometric loss ------
            GEOM_LOSS_TYPE="chamfer",  # L1, chamfer
            GEOM_LW=100.0,  # 100
            CHAMFER_CENTER_LW=0.0,
            CHAMFER_DIST_THR=0.5,
            # refiner-based loss --------------
            REFINE_LW=0.0,
            # xyz loss (init, ren)
            XYZ_INIT_REN_LOSS_TYPE="L1",  # L1 | CE_coor (for cls)
            XYZ_INIT_REN_LW=0.0,
            # point matching loss using pseudo pose ---------------------------
            SELF_PM_CFG=dict(
                loss_weight=10.0,  # NOTE: >0 to enable this loss
            ),
        ),
    ),
)

# gu
# objects  ape     Avg(1)
# ad_2     4.95    4.95
# ad_5     42.67   42.67
# ad_10    75.71   75.71
# rete_2   26.29   26.29
# rete_5   95.52   95.52
# rete_10  100.00  100.00
# re_2     26.29   26.29
# re_5     95.52   95.52
# re_10    100.00  100.00
# te_2     96.76   96.76
# te_5     100.00  100.00
# te_10    100.00  100.00
# proj_2   86.67   86.67
# proj_5   98.67   98.67
# proj_10  100.00  100.00
# re       2.83    2.83
# te       0.01    0.01
