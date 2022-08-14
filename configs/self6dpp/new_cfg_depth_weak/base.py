_base_ = ["../../_base_/self6dpp_base.py"]

OUTPUT_DIR = "YOU SHOULD REPLACE THIS"
INPUT = dict(
    WITH_DEPTH=True,
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=False,
    CHANGE_BG_PROB=0.5,  # 0.5
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="iaa_custom",
    COLOR_AUG_BG_REPLACE="datasets/VOCdevkit/VOC2012/JPEGImages",
    COLOR_AUG_CACHED_BG=False,
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
        OVERALL_PROB=0.2,
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
    IMS_PER_BATCH=4,  # 6, maybe need to be < 24
    TOTAL_EPOCHS=100,
    CHECKPOINT_PERIOD=30,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-6, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=100,  # NOTE: only real data, iterations are very small
    CLIP_GRADIENTS=dict(ENABLED=True, CLIP_TYPE="full_model", CLIP_VALUE=200),
    CILP_GRAD=500,
)

'''
Turn into GDRN_MaskNormVF

1, MODEL.POSE_NET.NAME
2, MODEL.GEO_HEAD.INIT_CFG.type
3, MODEL.PNP_NET.INIT_CFG.type
4, MODEL.PNP_NET.WITH_VF
5, MODEL.PNP_NET.MASK_ATTENTION
6, MODEL.PNP_NET.WITH_NORM
7, MODEL.PNP_NET.NORM_ATTENTION
'''

RENDERER = dict(
    ENABLE=False,
    DIFF_RENDERER="new_DIBR",
    COLOR_RANGE=255,  # fix a bug when using PLY+new_DIBR
)  # DIBR | dibr | new_DIBR
MODEL = dict(
    # synthetically trained model
    WEIGHTS="YOU SHOULD REPLACE THIS",
    # init
    # ad10    rete5    te2
    # 57.90   57.24    85.33
    REFINER_WEIGHTS="",
    FREEZE_BN=True,
    SELF_TRAIN=True,  # whether to do self-supervised training
    WITH_REFINER=False,  # whether to use refiner
    LOAD_DETS_TRAIN=True,  # NOTE: load detections for self-train
    LOAD_DETS_TRAIN_WITH_POSE=True,  # NOTE: load pose_refine
    LOAD_DETS_TEST=True,
    PSEUDO_POSE_TYPE="pose_refine",  # pose_est | pose_refine | pose_init (online inferred by teacher)
    EMA=dict(
        ENABLED=True,
        INIT_CFG=dict(decay=0.999, updates=0),  # epoch-based
        UPDATE_FREQ=10,  # update the mean teacher every n epochs
    ),
    POSE_NET=dict(
        NAME="GDRN_MaskNormVF",  # used module file name  GDRN_double_mask_double_vf
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
                type="TopDownMaskNormVFXyzRegionHead",  # TopDownDoubleMaskDoubleVFXyzRegionHead TopDownDoubleMaskXyzRegionHead
                in_dim=2048,  # this is num out channels of backbone conv feature
            ),
            NUM_REGIONS=64,
            VF_CLASS_AWARE=False,
            NUM_CHANNAL_VF=16,
            NORM_CLASS_AWARE=False,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(
                norm="GN",
                act="gelu",
                type="ConvPnPNetMaskNormVF",  # ConvPnPNet ConvPnPNetAll
            ),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            MASK_ATTENTION="concat",  # concat none
            WITH_VF="both",
            WITH_NORM="both",
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
            # vf loss ---------------------------
            VF_LOSS_TYPE="L1+Cos",
            VIS_VF_LW=1.0,
            FULL_VF_LW=1.0,
            # vf-rt loss ---------------------------
            VF_RT_LOSS_TYPE="L1+Cos",
            VF_RT_LW=1.0,
            # vertex norm loss ---------------------------
            NORM_LOSS_TYPE="L1+Cos",
            VIS_NORM_LW=1.0,
            FULL_NORM_LW=1.0,
            # vertex norm-rt loss ---------------------------
            NORM_RT_LOSS_TYPE="L1+Cos",
            NORM_RT_LW=1.0,
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
            # vf loss -------------------------
            VIS_VF_LW=1.0,
            FULL_VF_LW=1.0,
            # vf-rt loss
            VIS_RT_VF_LW=10.0,
            FULL_RT_VF_LW=10.0,
            # vertex norm loss ---------------------------
            VIS_NORM_LW=1.0,
            FULL_NORM_LW=1.0,
            # vertex norm-rt loss ---------------------------
            VIS_NORM_RT_LW=10.0,
            FULL_NORM_RT_LW=10.0,
            # LAB space loss ------------------
            LAB_NO_L=True,
            LAB_LW=0.,
            # MS-SSIM loss --------------------
            MS_SSIM_LW=1.0,
            # perceptual loss -----------------
            PERCEPT_LW=0.15,
            # mask loss (init, ren) -----------------------
            MASK_INIT_REN_LOSS_TYPE="RW_BCE",
            # MASK_INIT_REN_LOSS_TYPE="dice",
            MASK_INIT_REN_LW=1.0,
            # mask loss (init, pred) -----------------------
            MASK_INIT_PRED_LOSS_TYPE="vis+full",
            MASK_INIT_PRED_LW=1.0,
            # depth-based geometric loss ------
            GEOM_LOSS_TYPE="chamfer",  # chamfer
            GEOM_LW=0.0,
            CHAMFER_CENTER_LW=0.0,
            CHAMFER_DIST_THR=0.5,
            # refiner-based loss --------------
            REFINE_LW=0.0,
            # xyz loss (init, pred)
            XYZ_INIT_PRED_LOSS_TYPE="smoothL1",  # L1 | CE_coor (for cls)
            XYZ_INIT_PRED_LW=1.0,
            # point matching loss using pseudo pose ---------------------------
            SELF_PM_CFG=dict(
                loss_weight=10.0,  # NOTE: >0 to enable this loss
            ),
            # trans loss
            TRANS_LW=10.0,
        ),
    ),
)

TRAIN = dict(
    DEBUG_SINGLE_IM=False,
    RECORD_DURING_TRAINING=True,
)

TEST = dict(
    EVAL_PERIOD=3,  # count in epochs
    VIS=False,
    TEST_BBOX_TYPE="est"
)  # gt | est

REPJ_REFINE = dict(
    ENABLE=True,
    REPJ_REFINER_LW=dict(
        IOU2D3D=0.1,
        PM=10,
        MIOU=1,
        MSSSIM=1,
    ),
    RENDERER=dict(
        SHRINK=1,
    ),
    DISTANCE_INVERSE_SAMPLER=dict(
        ENABLE=True,
        PSEUDO_POSE_TYPE="pose_refine",  # pose_est | pose_refine
    )
)
