import random
import logging
from typing import Tuple
import os
import os.path as osp
import cv2
from tqdm import tqdm
import numpy as np
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from imgaug.augmenters.arithmetic import multiply_elementwise


logger = logging.getLogger(__name__)


class AugmentRGB(object):
    """Augmentation tool for detection problems.

    Parameters
    ----------

    brightness_var: float, default: 0.3
        The variance in brightness

    hue_delta: float, default: 0.1
        The variance in hue

    lighting_std: float, default: 0.3
        The standard deviation in lighting

    saturation_var: float, default: (0.5, 1.25)
        The variance in saturation

    contrast_var: float, default: (0.5, 1.25)
        The variance in constrast

    swap_colors: bool, default: False
        Whether color channels should be randomly flipped

    Notes
    -----
    Assumes images and labels to be in the range [0, 1] (i.e. normalized)

    All new operations are drafted from the TF implementation
    https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/python/ops/image_ops_impl.py

    Look here: https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_pascal_resnet.py
    """

    def __init__(
        self,
        brightness_delta=32.0 / 255.0,
        hue_delta=0,
        lighting_std=0.3,
        saturation_var=(0.75, 1.25),
        contrast_var=(0.75, 1.25),
        swap_colors=False,
    ):

        # Build a list of color jitter functions
        self.color_jitter = []

        if brightness_delta:
            self.brightness_delta = brightness_delta
            self.color_jitter.append(self.random_brightness)
        if hue_delta:
            self.hue_delta = hue_delta
            self.color_jitter.append(self.random_hue)
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.random_saturation)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.random_contrast)

        self.lighting_std = lighting_std
        self.swap_colors = swap_colors

    def augment(self, img):

        augment_type = np.random.randint(0, 2)
        if augment_type == 0:  # Take the image as a whole
            pass
        elif augment_type == 1:  # Random downsizing of original image
            pass  # img, lbl = self.random_rescale(img, lbl)

        random.shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            img = jitter(img)
        return img

    def random_brightness(self, img):
        """Adjust the brightness of images by a random factor.

        Basically consists of a constant added offset.
        Args:
          image: An image.
        Returns:
          The brightness-adjusted image.
        """
        max_delta = self.brightness_delta
        assert max_delta >= 0
        delta = -max_delta + 2 * np.random.rand() * max_delta
        return np.clip(img + delta, 0.0, 1.0)

    def random_contrast(self, img):
        """For each channel, this function computes the mean of the image
        pixels in the channel and then adjusts each component.

        `x` of each pixel to `(x - mean) * contrast_factor + mean`.
        Args:
          image: RGB image or images. Size of the last dimension must be 3.
        Returns:
          The contrast-adjusted image.
        """
        lower, upper = self.contrast_var
        assert 0.0 <= lower <= upper
        contrast_factor = lower + 2 * np.random.rand() * (upper - lower)
        means = np.mean(img, axis=(0, 1))
        return np.clip((img - means) * contrast_factor + means, 0.0, 1.0)

    def random_saturation(self, img):
        """Adjust the saturation of an RGB image by a random factor.

        Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
        picked in the interval `[lower, upper]`.
        Args:
          image: RGB image or images. Size of the last dimension must be 3.
        Returns:
          Adjusted image(s), same shape and DType as `image`.
        """
        lower, upper = self.saturation_var
        assert 0.0 <= lower <= upper
        saturation_factor = lower + 2 * np.random.rand() * (upper - lower)
        return self.adjust_saturation(img, saturation_factor)

    def random_hue(self, img):
        """Adjust the hue of an RGB image by a random factor.

        Equivalent to `adjust_hue()` but uses a `delta` randomly
        picked in the interval `[-max_delta, max_delta]`.
        `hue_delta` must be in the interval `[0, 0.5]`.
        Args:
          img: RGB image or images. Size of the last dimension must be 3.
        Returns:
          Numpy image
        """
        max_delta = self.hue_delta
        assert 0.0 <= max_delta <= 0.5
        delta = -max_delta + 2.0 * np.random.rand() * max_delta
        return self.adjust_hue(img, delta)

    def adjust_gamma(self, img, gamma=1.0, gain=1.0):
        """Performs Gamma Correction on the input image.
          Also known as Power Law Transform. This function transforms the
          input image pixelwise according to the equation Out = In**gamma
          after scaling each pixel to the range 0 to 1.
        Args:
          img : Numpy array.
          gamma : A scalar. Non negative real number.
          gain  : A scalar. The constant multiplier.
        Returns:
          Gamma corrected numpy image.
        Notes:
          For gamma greater than 1, the histogram will shift towards left and
          the output image will be darker than the input image.
          For gamma less than 1, the histogram will shift towards right and
          the output image will be brighter than the input image.
        References:
          [1] http://en.wikipedia.org/wiki/Gamma_correction
        """

        assert gamma >= 0.0
        # According to the definition of gamma correction
        return np.clip(((img ** gamma) * gain), 0.0, 1.0)

    def adjust_hue(self, img, delta):
        """Adjust hue of an RGB image.

        Converts an RGB image to HSV, add an offset to the hue channel and converts
        back to RGB. Rotating the hue channel (H) by `delta`.
        `delta` must be in the interval `[-1, 1]`.
        Args:
            image: RGB image
            delta: float.  How much to add to the hue channel.
        Returns:
            Adjusted image as np
        """
        assert img.shape[2] == 3
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # OpenCV returns hue from 0 to 360
        hue, sat, val = cv2.split(hsv)

        # Note that we add 360 to guarantee that the resulting hue is a positive
        # floating point number since delta is [-0.5, 0.5].
        hue = np.mod(360 + hue + delta * 360, 360.0)
        hsv_altered = cv2.merge((hue, sat, val))
        return cv2.cvtColor(hsv_altered, cv2.COLOR_HSV2BGR)

    def adjust_saturation(self, img, saturation_factor):
        """Adjust saturation of an RGB image.

        `image` is an RGB image.  The image saturation is adjusted by converting the
        image to HSV and multiplying the saturation (S) channel by
        `saturation_factor` and clipping. The image is then converted back to RGB.
        Args:
          img: RGB image or images. Size of the last dimension must be 3.
          saturation_factor: float. Factor to multiply the saturation by.
        Returns:
          Adjusted numpy image
        """

        assert img.shape[2] == 3
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        sat = np.clip(sat * saturation_factor, 0.0, 1.0)
        hsv_altered = cv2.merge((hue, sat, val))
        return cv2.cvtColor(hsv_altered, cv2.COLOR_HSV2BGR)

    def swap_colors(self, img):
        """Randomly swap color channels."""
        # Skip swapping?
        if np.random.random() > 0.5:
            return img

        swap = np.random.randint(5)
        if swap == 0:
            img = 1.0 - img
        elif swap == 1:
            img = img[:, :, [0, 2, 1]]
        elif swap == 2:
            img = img[:, :, [2, 0, 1]]
        elif swap == 3:
            img = img[:, :, [1, 0, 2]]
        elif swap == 4:
            img = img[:, :, [1, 2, 0]]
        return img

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        """Randomly change saturation."""
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 1)

    def brightness(self, rgb):
        """Randomly change brightness."""
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 1)

    def contrast(self, rgb):
        """Randomly change contrast levels."""
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 1)

    def lighting(self, img):
        """Randomly change lighting."""
        cov = np.cov(img.reshape(-1, 3), rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise)
        img += noise
        return np.clip(img, 0, 1)


class CoarseImgPatch(iaa.CoarseDropout):
    def __init__(
        self,
        p=(0.02, 0.1),
        size_px=None,
        size_percent=None,
        image_pth=None,
        cached_img=False,
        cached_img_limit=5000,
    ):
        super().__init__(
            p=p,
            size_px=size_px,
            size_percent=size_percent,
            per_channel=False,
            min_size=3,
            seed=None,
            name=None,
            random_state="deprecated",
            deterministic="deprecated",
        )
        self.__img_list = []
        for obj in os.listdir(image_pth):
            obj = osp.join(osp.abspath(image_pth), obj)
            if osp.isfile(obj) and osp.splitext(obj)[1] in {".png", ".jpg", "jpeg"}:
                self.__img_list.append(obj)

        np.random.shuffle(self.__img_list)
        self.__img_list = self.__img_list[: cached_img_limit]
        img_cnt = len(self.__img_list)
        assert img_cnt, image_pth

        self.__img_cache = []
        if cached_img:
            logger.warning(
                "You are trying to preload all {} backgrounds which will be used during color_aug::CoarseImgPatch. "
                "You can set cfg.INPUT.COLOR_AUG_CACHED_BG=False to disable this feature.".format(
                    img_cnt
                )
            )
            size_sum = 0
            for img_dir in tqdm(self.__img_list):
                img: np.ndarray = cv2.imread(img_dir).astype(np.uint8)
                self.__img_cache.append(img)
                h, w, c = img.shape
                size_sum += h * w * c
            logger.warning(
                "{} background images are cached, take approx {:.2f} GiB".format(
                    img_cnt, size_sum / 1024 ** 3
                )
            )

        self.__img_id_picker = iap.Choice(list(range(img_cnt)))
        self.__random_crop_gen = iap.Uniform(0, 1)

    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images
        nb_images = len(images)
        rss = random_state.duplicate(1 + nb_images)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[0]
        )
        is_mul_binomial = isinstance(self.mul, iap.Binomial) or (
            isinstance(self.mul, iap.FromLowerResolution)
            and isinstance(self.mul.other_param, iap.Binomial)
        )

        gen = enumerate(zip(images, per_channel_samples, rss[1:]))
        for i, (image, per_channel_samples_i, rs) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (
                height,
                width,
                nb_channels if per_channel_samples_i > 0.5 else 1,
            )
            mask = self.mul.draw_samples(sample_shape, random_state=rs)
            inv_mask = 1 - mask

            img_id = self.__img_id_picker.draw_samples(1, random_state=rs)[0]
            bg_ori_img = self._get_bg(img_id)
            background = self._background_img_preprocess(
                bg_ori_img, mask.shape, random_state=rs
            )

            if mask.dtype.kind != "b" and is_mul_binomial:
                mask = mask.astype(bool, copy=False)
                inv_mask = inv_mask.astype(bool, copy=False)

            batch.images[i] = multiply_elementwise(image, mask) + multiply_elementwise(
                background, inv_mask
            )

        return batch

    def _get_bg(self, img_id: int):
        if len(self.__img_cache) == 0:
            img_dir = self.__img_list[img_id]
            return cv2.imread(img_dir)
        else:
            return self.__img_cache[img_id]

    def _background_img_preprocess(
        self, img: np.ndarray, align_size: Tuple[int, int, int], random_state
    ):
        align_h, align_w, _ = align_size
        ori_h, ori_w, _ = img.shape
        h_ratio, w_ratio = align_h / ori_h, align_w / ori_w
        larger_ratio = max(h_ratio, w_ratio)
        out_h = max(int(ori_h * larger_ratio), align_h)
        out_w = max(int(ori_w * larger_ratio), align_w)
        res = cv2.resize(img, dsize=(out_w, out_h))
        enlargement = self.__random_crop_gen.draw_samples(2, random_state=random_state)
        t_space = int((out_h - align_h) * enlargement[0])
        l_space = int((out_w - align_w) * enlargement[1])
        return res[t_space : t_space + align_h, l_space : l_space + align_w, :]


def build_iaa_color_augmenter(bg_replace_pth=None, cached_img=False):
    seq = iaa.Sequential(
        [
            iaa.Sometimes(
                0.5,
                CoarseImgPatch(
                    p=0.2,
                    size_percent=0.05,
                    image_pth=bg_replace_pth,
                    cached_img=cached_img,
                ),
            ),
            iaa.Sometimes(0.5, iaa.GaussianBlur(1.2 * np.random.rand())),
            iaa.Sometimes(0.3, iaa.pillike.EnhanceSharpness(factor=(0.0, 50.0))),
            iaa.Sometimes(0.3, iaa.pillike.EnhanceContrast(factor=(0.2, 50.0))),
            iaa.Sometimes(0.5, iaa.pillike.EnhanceBrightness(factor=(0.1, 6.0))),
            iaa.Sometimes(0.3, iaa.pillike.EnhanceColor(factor=(0.0, 20.0))),
            iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
            iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.2), per_channel=0.3)),
        ],
        random_order=True,
    )
    return seq
