from typing import List, Tuple
from detectron2.data import transforms as T
import numpy as np
import imgaug.augmenters as iaa
import logging

logger = logging.getLogger(__name__)


class RandomZoom(T.Augmentation):
    def __init__(self, scale_range: Tuple[float, float], keep_size=True):
        super(RandomZoom, self).__init__()
        self.scale_range = sorted(scale_range)
        self.keep_size = keep_size

    def get_transform(self, image) -> T.Transform:
        old_h, old_w = image.shape[-2:]
        scale_factor = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]

        new_h, new_w = int(old_h * scale_factor), int(old_w * scale_factor)
        ch, cw = int(new_h // 2), int(new_w // 2)
        sh, sw = int(ch / scale_factor), int(cw / scale_factor)

        if self.keep_size:
            return T.TransformList([
                T.ResizeTransform(old_h, old_w, new_h, new_w),
                T.CropTransform(ch, cw, sh, sw, old_h, old_w)
            ])
        else:
            return T.TransformList([
                T.ResizeTransform(old_h, old_w, new_h, new_w),
                T.CropTransform(ch, cw, sh, sw)
            ])


def build_gdrn_augmentation_pose_variated(cfg, is_train: bool) -> iaa.Augmenter:
    """Create a list of :class:`Augmentation` from config.

    Returns:
        list[Augmentation]
    """
    if not is_train:
        return iaa.Noop()

    aug_cfg = cfg.INPUT.POSE_VARIATED_AUG
    rot_deg = aug_cfg.ROT.MAX_DEGREE // 2
    sl, su = aug_cfg.ZOOM.LOWER_LIM, aug_cfg.ZOOM.UPPER_LIM
    tl, tu = aug_cfg.TRANS.LOWER_LIM, aug_cfg.TRANS.UPPER_LIM
    crop_percent = aug_cfg.CROP.PERCENT
    seq = iaa.Sequential(
        [
            iaa.CropAndPad(
                percent=(-crop_percent, crop_percent),
                sample_independently=False,
                pad_mode="constant",
            ),
            iaa.Affine(
                scale={"x": (sl, su), "y": (sl, su)},
                translate_percent={"x": (tl, tu), "y": (tl, tu)},
                rotate=(-rot_deg, rot_deg),
            ),
        ],
        random_order=True,
    )

    return seq


def build_gdrn_augmentation(cfg, is_train):
    """Create a list of :class:`Augmentation` from config. when training 6d
    pose, cannot flip.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        # augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation