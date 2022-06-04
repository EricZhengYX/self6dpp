import os
import os.path as osp
import warnings


def validate_lm_single_seq(ds_pth: str, model_pth: str, obj_idx: int):
    sub_pth = {
        "depth",
        "mask",
        "mask_visib",
        "rgb",
    }
    item_cnt_consist = set()
    for suffix in sub_pth:
        _p = osp.join(ds_pth, suffix)
        assert osp.exists(_p), "The {s} folder of seq {i} dose not exist.".format(
            s=suffix, i=obj_idx
        )
        _cnt = 0
        for f in os.listdir(_p):
            if osp.isfile(f):
                _cnt += 1
        item_cnt_consist.add(_cnt)
    assert (
        len(item_cnt_consist) == 1
    ), "The numbers of items are not consist in seq {}".format(obj_idx)
    if not osp.exists(osp.join(ds_pth, "cam_para_gt.json")) and not osp.exists(
        osp.join(ds_pth, "cam_para_gt.pkl")
    ):
        warnings.warn("Missing cam_para_gt in seq {}".format(obj_idx))
    # if not osp.exists(osp.join(ds_pth, 'pose_inf.json')):
    #     warnings.warn('Missing pose_inf.json in seq {}'.format(obj_idx))
    assert osp.exists(
        osp.join(ds_pth, "scene_gt.json")
    ), "Missing scene_gt.json in seq {}".format(obj_idx)
    assert osp.exists(model_pth), "Missing model in seq {}".format(obj_idx)


def validate_lm(ds_root_pth: str, model_root_pth: str, seq: list):
    assert osp.exists(
        ds_root_pth
    ), "The root path of LM dataset dose not exist. {}".format(ds_root_pth)
    assert osp.exists(
        model_root_pth
    ), "The root path of LM models dose not exist. {}".format(model_root_pth)
    models_info_json_pth = osp.join(model_root_pth, "models_info.json")
    assert osp.exists(
        models_info_json_pth
    ), "The size info of models dose not exist. {}".format(models_info_json_pth)
    for i in seq:
        _p_ds = osp.join(ds_root_pth, str(i).zfill(6))
        _p_model = osp.join(model_root_pth, "obj_{}.ply".format(str(i).zfill(6)))
        assert osp.exists(_p_ds), "The folder of seq {} dose not exist.".format(i)
        assert osp.exists(_p_model), "The model of seq {} dose not exist.".format(i)
        assert _p_model[-4:] == ".ply", "Only support ply"
        validate_lm_single_seq(_p_ds, _p_model, i)
