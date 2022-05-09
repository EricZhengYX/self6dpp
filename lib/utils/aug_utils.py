from typing import List, Tuple
from detectron2.data import transforms as T
import numpy as np

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
