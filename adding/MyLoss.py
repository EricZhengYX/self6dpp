import torch
import torch.nn as nn
import numpy as np
from chamferdist import ChamferDistance

HEIGHT = 120
WIDTH = 160
K = torch.tensor([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

class SelfLoss(nn.Module):

    def __init__(self, with_cham_loss=True, with_mask_loss=True, HEIGHT=HEIGHT, WIDTH=WIDTH, K=K, device='cuda'):
        super().__init__()

        self.H, self.W = HEIGHT, WIDTH
        self.K = K
        self.device = device

        self.with_cham_loss = with_cham_loss
        if self.with_cham_loss:
            self.chamferDist = ChamferDistance()
        self.with_mask_loss = with_mask_loss

    def forward(self, pred_PM, pred_Ms, depth_src=None, depth_tgt=None):
        """

        Args:
            pred_PM: PM of source mask, 1HW
            pred_Ms: mask(binary) of target mask, 1HW
            depth_src: source depth (for chamfer loss only), HW
            depth_tgt: target depth (for chamfer loss only), HW

        Returns:
            mask loss + chamfer loss

        """
        WEIGHT_CHAMFER = 1
        WEIGHT_MASK = 1

        self.__check_masks(pred_PM, pred_Ms)
        cham_loss_sum = torch.tensor(0.0, device=self.device)
        mask_loss_sum = torch.tensor(0.0, device=self.device)
        if self.with_cham_loss:
            assert isinstance(depth_src, torch.Tensor)
            assert isinstance(depth_tgt, torch.Tensor)

            src_pc_bp = self.backproject_torch(depth_src, K)  # point cloud (from a backprojection, for the src)
            src_points_bp = src_pc_bp[src_pc_bp[:, :, 2] > 0]  # Nx3

            tgt_pc_bp = self.backproject_torch(depth_tgt, K)  # point cloud (from a backprojection, for the tgt)
            tgt_points_bp = tgt_pc_bp[tgt_pc_bp[:, :, 2] > 0]  # Nx3

            cham_loss = self.chamferDist(
                src_points_bp.unsqueeze(0),
                tgt_points_bp.unsqueeze(0),
                # bidirectional=True
            )
            cham_loss_sum = cham_loss_sum + WEIGHT_CHAMFER * cham_loss

        if self.with_mask_loss:
            mask_edge_weights = self.compute_mask_edge_weights(
                pred_Ms.view(-1, 1, self.H, self.W),
                dilate_kernel_size=9,
                erode_kernel_size=9)
            loss_mask = self.weighted_ex_loss_probs(
                pred_PM.view(-1, 1, self.H, self.W),
                pred_Ms.view(-1, 1, self.H, self.W),
                mask_edge_weights)
            mask_loss_sum = mask_loss_sum + WEIGHT_MASK * loss_mask

        return cham_loss_sum, mask_loss_sum

    def __check_masks(self, mask_s, mask_t):
        assert isinstance(mask_s, torch.Tensor) and isinstance(mask_t, torch.Tensor)
        assert mask_s.shape == mask_t.shape
        assert mask_t.device == mask_s.device

    def get_points_diameter(self, points):
        minx, maxx = points[:, 0].min(), points[:, 0].max()
        miny, maxy = points[:, 1].min(), points[:, 1].max()
        minz, maxz = points[:, 2].min(), points[:, 2].max()

        return torch.tensor([minx - maxx, miny - maxy, minz - maxz]).norm()

    # def remove_outliers_statistical(self, points, std_ratio=1.5):
    #     try:
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(points)
    #
    #         cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=std_ratio)
    #         return np.asarray(pcd.select_down_sample(ind).points)
    #     except:
    #         return points

    def points_ren_real_bp_chamfer_loss(self, ren_depths, real_depths, Ks, rois, distance_threshold=0.05):
        """
        NOTE:
            ren_depths: BHW
            real_depths: BHW
            target points: depth(masked) => backproject (K)
        """
        bs = len(ren_depths)
        num_valid = 0
        loss = torch.tensor(0.0).to(ren_depths)
        for i in range(bs):
            if Ks.ndim == 2:
                K = Ks
            else:
                K = Ks[i]

            real_pc_bp = self.backproject_torch(real_depths[i], K)
            real_points_bp = real_pc_bp[real_pc_bp[:, :, 2] > 0]  # Nx3

            rend_pc_bp = self.backproject_torch(ren_depths[i], K)
            rend_points_bp = rend_pc_bp[rend_pc_bp[:, :, 2] > 0]  # Nx3

            if torch.cuda.is_available():
                real_points_bp = real_points_bp.cuda()

            dist1, dist2 = self.chamferdistmodule(real_points_bp[None], rend_points_bp[None])

            cur_loss = torch.mean(dist1) + torch.mean(dist2)

            if torch.isnan(cur_loss) or cur_loss <= 0.:
                continue

            loss += cur_loss
            num_valid += 1

        return loss / max(num_valid, 1)

    def weighted_ex_loss_probs(self, probs, target, weight):
        """
        https://github.com/PengtaoJiang/OAA-PyTorch
        http://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Integral_Object_Mining_via_Online_Attention_Accumulation_ICCV_2019_paper.pdf
        """
        assert probs.size() == target.size()
        pos = torch.gt(target, 0)
        neg = torch.eq(target, 0)
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)
        pos_loss = -target[pos] * torch.log(probs[pos]) * weight[pos]
        neg_loss = -torch.log(1 - probs[neg]) * weight[neg]
        if torch.isnan(pos_loss).any():
            # print('pos_loss', pos_loss)
            print("pos_loss nan", target.min(), target.max(), probs.min(), probs.max())
        if torch.isnan(neg_loss).any():
            # print('neg_loss', neg_loss)
            print("neg_loss nan", target.min(), target.max(), probs.min(), probs.max())

        loss = 0.0
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        if num_pos > 0:
            loss += 1.0 / num_pos.float() * torch.sum(pos_loss)
        if num_neg > 0:
            loss += 1.0 / num_neg.float() * torch.sum(neg_loss)

        return loss

    def project_torch(self, points_3d, K):
        points_2d = (K @ points_3d.T).T
        return points_2d[:, :2] / points_2d[:, 2:3]


    def backproject_torch(self, depth, K):
        """ Backproject a depth map to a cloud map
        :param depth: Input depth map [H, W]
        :param K: Intrinsics of the camera
        :return: An organized cloud map
        """
        import torch
        assert depth.ndim == 2, depth.ndim
        H, W = depth.shape[:2]
        X = torch.tensor(range(W)).to(depth) - K[0, 2]
        X = X.repeat(H, 1)
        Y = torch.tensor(range(H)).to(depth) - K[1, 2]
        Y = Y.repeat(W, 1).t()
        return torch.stack((X * depth / K[0, 0], Y * depth / K[1, 1], depth), dim=2)


    def compute_mask_edge_weights(self, mask, dilate_kernel_size=5, erode_kernel_size=5, w_edge=5.0, edge_lower=True):
        """defined in Contour Loss: Boundary-Aware Learning for Salient Object Segmentation
        (https://arxiv.org/abs/1908.01975)
        mask: [B, 1, H, W]
        """
        dilated_mask = self.mask_dilate_torch(mask, kernel_size=dilate_kernel_size)
        eroded_mask = self.mask_dilate_torch(mask, kernel_size=erode_kernel_size)
        # edge width: kd//2 + ke//2 ?
        mask_edge = dilated_mask - eroded_mask
        # old (bug, edge has lower weight)
        if edge_lower:
            # >1 for non-edge, ~1 for edge
            return torch.exp(-0.5 * (mask_edge * w_edge) ** 2) / (np.sqrt(2 * np.pi)) + 1
        else:
            # 1 for non-edge, >1 for edge
            _gaussian = torch.exp(-0.5 * (mask_edge * w_edge) ** 2) / (np.sqrt(2 * np.pi))
            return _gaussian.max() - _gaussian + 1  # new

    def mask_dilate_torch(self, mask, kernel_size=3):
        """
        mask: [B,1,H,W]
        """
        if isinstance(kernel_size, (int, float)):
            kernel_size = (int(kernel_size), int(kernel_size))
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        kernel = torch.ones(kernel_size)[None, None].to(mask)
        result = torch.clamp(nn.functional.conv2d(mask, kernel, padding=padding), 0, 1)
        return result
