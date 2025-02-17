# differentiable renderer utils
import cv2
import numpy as np
import torch
import os
import os.path as osp
from tqdm import tqdm

# from kaolin.rep import TriangleMesh
# from kaolin.graphics import DIBRenderer
from .rep import TriangleMesh
from .dib_renderer_x import DIBRenderer
from adding.DIB_extended.utils.pose_utils import quat2mat_torch

OBJ_IN_MM = True
K = torch.tensor([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

def transform_pts_Rt_th(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 tensor with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 tensor with transformed 3D points.
    """
    assert pts.shape[1] == 3
    if not isinstance(pts, torch.Tensor):
        pts = torch.as_tensor(pts)
    if not isinstance(R, torch.Tensor):
        R = torch.as_tensor(R).to(pts)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t).to(pts)
    pts_t = torch.matmul(R, pts.t()) + t.view(3, 1)
    return pts_t.t()


def load_objs(obj_paths, texture_paths=None, height=480, width=640):
    if isinstance(obj_paths, str):
        obj_paths = [obj_paths]
    if isinstance(texture_paths, str):
        texture_paths = [texture_paths]

    assert all([".obj" in _path for _path in obj_paths])
    if texture_paths is not None:
        assert len(obj_paths) == len(texture_paths)
    models = []
    for i, obj_path in enumerate(tqdm(obj_paths)):

        model = {}
        mesh = TriangleMesh.from_obj(obj_path)

        vertices = mesh.vertices[:, :3]  # x,y,z
        colors = mesh.vertices[:, 3:6]  # rgb
        faces = mesh.faces.int()
        ###########################
        # normalize verts ( - center)  ！！！不需要！！！
        ###########################
        # vertices_max = vertices.max()
        # vertices_min = vertices.min()
        # vertices_middle = (vertices_max + vertices_min) / 2.0
        # vertices = (vertices - vertices_middle) / (1000 if OBJ_IN_MM else 1)
        vertices = vertices / (1000 if OBJ_IN_MM else 1)
        model["vertices"] = vertices[None, :, :].cuda()

        ###########################
        # Generate vertex color
        ###########################
        if colors.shape[1] == 0:
            vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
            vert_max = torch.max(vertices, dim=1, keepdims=True)[0]
            model["colors"] = ((vertices - vert_min) / (vert_max - vert_min))[None, :, :].cuda()
        else:
            model["colors"] = colors[None, :, :].cuda()

        model["faces"] = faces[None, :, :].cuda()  # NOTE: -1
        if texture_paths is not None:
            uvs = mesh.uvs
            face_textures = mesh.face_textures  # NOTE: -1
            assert osp.exists(texture_paths[i]), texture_paths[i]
            texture = cv2.imread(texture_paths[i], cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
            texture = cv2.resize(texture, (width, height), interpolation=cv2.INTER_AREA)
            # print('texture map: ', texture.shape)
            texture = torch.from_numpy(texture.transpose(2, 0, 1)[None, :, :, :]).cuda()
            model["uvs"] = uvs[None, :, :].cuda()
            model["face_textures"] = face_textures[None, :, :].cuda()
            model["texture"] = texture.cuda()
        models.append(model)
    return models


def load_plys(ply_paths, height=480, width=640, device='cuda'):
    if isinstance(ply_paths, str):
        ply_paths = [ply_paths]

    assert all([".ply" in _path for _path in ply_paths])
    models = []
    for i, ply_path in enumerate(tqdm(ply_paths)):

        model = {}
        mesh = TriangleMesh.from_ply(ply_path)
        vertices = mesh.vertices[:, :3]  # x,y,z
        colors = mesh.vertices[:, 3:6]  # rgb
        faces = mesh.faces.int()
        ###########################
        # normalize verts ( - center)  ！！！不需要！！！
        ###########################
        # vertices_max = vertices.max()
        # vertices_min = vertices.min()
        # vertices_middle = (vertices_max + vertices_min) / 2.0
        # vertices = (vertices - vertices_middle) / (1000 if OBJ_IN_MM else 1)
        vertices = vertices / (1000 if OBJ_IN_MM else 1)
        model["vertices"] = vertices[None, :, :].to(device)

        ###########################
        # Generate vertex color
        ###########################
        if colors.shape[1] == 0:
            vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
            vert_max = torch.max(vertices, dim=1, keepdims=True)[0]
            model["colors"] = ((vertices - vert_min) / (vert_max - vert_min))[None, :, :].to(device)
        else:
            model["colors"] = colors[None, :, :].to(device)

        model["faces"] = faces[None, :, :].to(device) # NOTE: -1

        model["uvs"] = None
        model["face_textures"] = None
        model["texture"] = None

        models.append(model)
    return models


# def render_dib_vc_batch_single_model(
#     ren, Rs, ts, model, Ks=K, rot_type="mat", H=480, W=640, near=0.01, far=100.0, with_depth=True
# ):
#     """
#     Args:
#         ren: A DIB-renderer
#         model: A model loaded by load_plys
#     """
#     assert ren.mode in ["VertexColor"], ren.mode
#     assert len(Rs) == len(ts)
#     bs = len(Rs)
#
#     ren.set_camera_parameters_from_RT_K(Rs, ts.squeeze(), K, height=H, width=W, near=near, far=far, rot_type=rot_type)
#     if isinstance(model, dict):
#         pass
#     elif isinstance(model, list):
#         assert len(model) == 1, 'You are using single model mode but input multiple models.'
#         model = model[0]
#     else:
#         raise TypeError('"model" should be either dict or list.')
#     colors = torch.vstack([model["colors"] for _ in range(bs)])  # tensor b x p x 3
#     points = [torch.vstack([model["vertices"] for _ in range(bs)]),
#               model["faces"].long().squeeze()]  # [tensor b x p x 3, tensor f x 3]
#
#     # points: list of [vertices, faces]
#     # colors: list of colors
#     predictions, im_probs, _, im_masks = ren.forward(points=points, colors=colors)
#     if with_depth:
#         # transform xyz
#         if not isinstance(Rs, torch.Tensor):
#             Rs = torch.stack(Rs)  # list
#         if rot_type == "quat":
#             R_mats = quat2mat_torch(Rs)
#         else:
#             R_mats = Rs
#         # xyzs = [
#         #
#         #     transform_pts_Rt_th(model["vertices"][i], R_mats[i], ts[i])[None]
#         #     for i in range(bs)
#         # ]
#         xyzs = transform_pts_Rt_th(model["vertices"][0], R_mats[0], ts[0])[None]
#         ren_xyzs, _, _, _ = ren.forward(points=points, colors=xyzs)
#         depth = ren_xyzs[:, :, :, 2]  # bhw
#     else:
#         depth = None
#     # bxhxwx3 rgb, bhw1 prob, bhw1 mask, bhw depth
#     return predictions, im_probs, im_masks, depth


def render_dib_vc_batch_single_model(
    ren, Rs, ts, model, Ks=K, rot_type="mat", H=480, W=640, near=0.01, far=100.0, with_depth=True
):
    """
    Args:
        ren: A DIB-renderer
        model: A model loaded by load_plys
    """
    predictions, im_probs, im_masks, depth = render_dib_vc_batch(
    ren, Rs, ts, Ks, [0], [model], rot_type=rot_type, H=H, W=W, near=near, far=far, with_depth=with_depth)
    return predictions, im_probs, im_masks, depth


def render_dib_vc_batch(
    ren, Rs, ts, Ks, obj_ids, models, rot_type="quat", H=480, W=640, near=0.01, far=100.0, with_depth=False
):
    """
    Args:
        ren: A DIB-renderer
        models: All models loaded by load_objs, [dict, ...]
    """
    assert ren.mode in ["VertexColorBatch"], ren.mode
    bs = len(Rs)
    if len(Ks) == 1:
        Ks = [Ks[0] for _ in range(bs)]
    if isinstance(obj_ids, int):
        obj_ids = [obj_ids for _ in range(bs)]
    elif hasattr(obj_ids, '__iter__') and len(obj_ids) == 1:
        obj_ids = [obj_ids[0] for _ in range(bs)]
    elif obj_ids is None:
        obj_ids = list(range(bs))
    else:
        raise ValueError(obj_ids)

    ren.set_camera_parameters_from_RT_K(Rs, ts, Ks, height=H, width=W, near=near, far=far, rot_type=rot_type)
    colors = [models[_id]["colors"] for _id in obj_ids]  # b x [1, p, 3]
    points = [[models[_id]["vertices"], models[_id]["faces"][0].long()] for _id in obj_ids]

    # points: list of [vertices, faces]
    # colors: list of colors
    predictions, im_probs, _, im_masks = ren.forward(points=points, colors=colors)
    if with_depth:
        # transform xyz
        if not isinstance(Rs, torch.Tensor):
            Rs = torch.stack(Rs)  # list
        if rot_type == "quat":
            R_mats = quat2mat_torch(Rs)
        else:
            R_mats = Rs
        xyzs = [
            transform_pts_Rt_th(models[obj_id]["vertices"][0], R_mats[_id], ts[_id])[None]
            for _id, obj_id in enumerate(obj_ids)
        ]
        ren_xyzs, _, _, _ = ren.forward(points=points, colors=xyzs)
        depth = ren_xyzs[:, :, :, 2]  # bhw
    else:
        depth = None
    # bxhxwx3 rgb, bhw1 prob, bhw1 mask, bhw depth
    return predictions, im_probs, im_masks, depth


def render_dib_tex_batch(
    ren, Rs, ts, Ks, obj_ids, models, rot_type="quat", H=480, W=640, near=0.01, far=100.0, with_depth=False
):
    assert ren.mode in ["TextureBatch"], ren.mode
    bs = len(Rs)
    if len(Ks) == 1:
        Ks = [Ks[0] for _ in range(bs)]
    ren.set_camera_parameters_from_RT_K(Rs, ts, Ks, height=H, width=W, near=near, far=far, rot_type=rot_type)
    # points: list of [vertices, faces]
    points = [[models[_id]["vertices"], models[_id]["faces"][0].long()] for _id in obj_ids]
    uv_bxpx2 = [models[_id]["uvs"] for _id in obj_ids]
    texture_bx3xthxtw = [models[_id]["texture"] for _id in obj_ids]
    ft_fx3_list = [models[_id]["face_textures"][0] for _id in obj_ids]

    # points: list of [vertices, faces]
    # colors: list of colors
    dib_ren_im, dib_ren_prob, _, dib_ren_mask = ren.forward(
        points=points, uv_bxpx2=uv_bxpx2, texture_bx3xthxtw=texture_bx3xthxtw, ft_fx3=ft_fx3_list
    )

    if with_depth:
        # transform xyz
        if not isinstance(Rs, torch.Tensor):
            Rs = torch.stack(Rs)  # list
        if rot_type == "quat":
            R_mats = quat2mat_torch(Rs)
        else:
            R_mats = Rs
        xyzs = [
            transform_pts_Rt_th(models[obj_id]["vertices"][0], R_mats[_id], ts[_id])[None]
            for _id, obj_id in enumerate(obj_ids)
        ]
        dib_ren_vc_batch = DIBRenderer(height=H, width=W, mode="VertexColorBatch")
        dib_ren_vc_batch.set_camera_parameters(ren.camera_params)
        ren_xyzs, _, _, _ = dib_ren_vc_batch.forward(points=points, colors=xyzs)
        depth = ren_xyzs[:, :, :, 2]  # bhw
    else:
        depth = None
    return dib_ren_im, dib_ren_prob, dib_ren_mask, depth  # bxhxwx3 rgb, bhw1 prob/mask, bhw depth


def render_dib_vc_multi(ren, Rs, ts, K, obj_ids, models, rot_type="quat", H=480, W=640, near=0.01, far=100.0):
    assert ren.mode in ["VertexColorMulti"], ren.mode
    ren.set_camera_parameters_from_RT_K(Rs, ts, K, height=H, width=W, near=near, far=far, rot_type=rot_type)
    colors = [models[_id]["colors"] for _id in obj_ids]  # b x [1, p, 3]
    points = [[models[_id]["vertices"], models[_id]["faces"][0].long()] for _id in obj_ids]

    # points: list of [vertices, faces]
    # colors: list of colors
    predictions, im_prob, _, im_mask = ren.forward(points=points, colors=colors)
    # TODO: add depth
    return predictions, im_prob, im_mask  # 1xhxwx3 rgb


def render_dib_tex_multi(ren, Rs, ts, K, obj_ids, models, rot_type="quat", H=480, W=640, near=0.01, far=100.0):
    assert ren.mode in ["TextureMulti"], ren.mode
    ren.set_camera_parameters_from_RT_K(Rs, ts, K, height=H, width=W, near=near, far=far, rot_type=rot_type)
    # points: list of [vertices, faces]
    points = [[models[_id]["vertices"], models[_id]["faces"][0].long()] for _id in obj_ids]
    uv_bxpx2 = [models[_id]["uvs"] for _id in obj_ids]
    texture_bx3xthxtw = [models[_id]["texture"] for _id in obj_ids]
    ft_fx3_list = [models[_id]["face_textures"][0] for _id in obj_ids]

    dib_ren_im, dib_ren_prob, _, dib_ren_mask = ren.forward(
        points=points, uv_bxpx2=uv_bxpx2, texture_bx3xthxtw=texture_bx3xthxtw, ts=ts, ft_fx3=ft_fx3_list
    )
    # TODO: add depth
    return dib_ren_im, dib_ren_prob, dib_ren_mask  # 1xhxwx3 rgb, (1,h,w,1) prob/mask
