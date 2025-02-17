# Adapted from https://github.com/google-research/omniglue/blob/main/src/omniglue/dino_extract.py
# with added batch processing support

import cv2
import numpy as np
from third_party.dinov2 import dino
import tensorflow as tf
import torch


class DINOExtract:
    """Class to initialize DINO model and extract features from an image."""

    def __init__(self, cpt_path: str, feature_layer: int = 1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_layer = feature_layer
        self.model = dino.vit_base()
        state_dict_raw = torch.load(cpt_path, map_location='cpu')

        self.model.load_state_dict(state_dict_raw)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.cuda()
        self.model.half() 

        self.image_size_max = 630

        self.h_down_rate = self.model.patch_embed.patch_size[0]
        self.w_down_rate = self.model.patch_embed.patch_size[1]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.forward(image)

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        Extracts features from input images.

        Args:
            images (np.ndarray): Input images as a NumPy array of shape (N, H, W, C).

        Returns:
            np.ndarray: Extracted features with shape (N, H', W', D).
        """
        # Resize and process each input image
        image = [self._resize_input_image(img) for img in images]
        image_processed = torch.stack([self._process_image(img) for img in image])
        image_processed = image_processed.to(self.device).half()
        features = self.extract_feature(image_processed)
        features = features.permute(0, 2, 3, 1).cpu().numpy()
        return features

    def _resize_input_image(
        self, image: np.ndarray, interpolation=cv2.INTER_LINEAR
    ):
      """Resizes image such that both dimensions are divisble by down_rate."""
      h_image, w_image = image.shape[:2]
      h_larger_flag = h_image > w_image
      large_side_image = max(h_image, w_image)

      # resize the image with the largest side length smaller than a threshold
      # to accelerate ViT backbone inference (which has quadratic complexity).
      if large_side_image > self.image_size_max:
        if h_larger_flag:
          h_image_target = self.image_size_max
          w_image_target = int(self.image_size_max * w_image / h_image)
        else:
          w_image_target = self.image_size_max
          h_image_target = int(self.image_size_max * h_image / w_image)
      else:
        h_image_target = h_image
        w_image_target = w_image

      h, w = (
          h_image_target // self.h_down_rate,
          w_image_target // self.w_down_rate,
      )
      h_resize, w_resize = h * self.h_down_rate, w * self.w_down_rate
      image = cv2.resize(image, (w_resize, h_resize), interpolation=interpolation)
      return image

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
      """Turn image into pytorch tensor and normalize it."""
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])

      image_processed = image / 255.0
      image_processed = (image_processed - mean) / std
      image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)
      return image_processed

    @torch.no_grad()
    def extract_feature(self, image):
      """Extracts features from image.

      Args:
        image: (B, 3, H, W) torch tensor, normalized with ImageNet mean/std.

      Returns:
        features: (B, C, H//14, W//14) torch tensor image features.
      """
      b, _, h_origin, w_origin = image.shape
      out = self.model.get_intermediate_layers(image, n=self.feature_layer)[0]
      h = int(h_origin / self.h_down_rate)
      w = int(w_origin / self.w_down_rate)
      dim = out.shape[-1]
      out = out.reshape(b, h, w, dim).permute(0, 3, 1, 2).detach()
      return out

def _preprocess_shape(
    h_image, w_image, image_size_max=630, h_down_rate=14, w_down_rate=14
):
  # Flatten the tensors
  h_image = tf.squeeze(h_image)
  w_image = tf.squeeze(w_image)
  # logging.info(h_image, w_image)

  h_larger_flag = tf.greater(h_image, w_image)
  large_side_image = tf.maximum(h_image, w_image)

  # Function to calculate new dimensions when height is larger
  def resize_h_larger():
    h_image_target = image_size_max
    w_image_target = tf.cast(image_size_max * w_image / h_image, tf.int32)
    return h_image_target, w_image_target

  # Function to calculate new dimensions when width is larger or equal
  def resize_w_larger_or_equal():
    w_image_target = image_size_max
    h_image_target = tf.cast(image_size_max * h_image / w_image, tf.int32)
    return h_image_target, w_image_target

  # Function to keep original dimensions
  def keep_original():
    return h_image, w_image

  h_image_target, w_image_target = tf.cond(
      tf.greater(large_side_image, image_size_max),
      lambda: tf.cond(h_larger_flag, resize_h_larger, resize_w_larger_or_equal),
      keep_original,
  )

  # resize to be divided by patch size
  h = h_image_target // h_down_rate
  w = w_image_target // w_down_rate
  h_resize = h * h_down_rate
  w_resize = w * w_down_rate

  # Expand dimensions
  h_resize = tf.expand_dims(h_resize, 0)
  w_resize = tf.expand_dims(w_resize, 0)

  return h_resize, w_resize

def get_dino_descriptors(dino_features, keypoints, height, width, feature_dim):
    """Modified to support batch processing

    Extracts DINO descriptors at provided keypoints using bilinear interpolation in TensorFlow.

    Args:
        dino_features (tf.Tensor): The feature map tensor from the DINO model.
        keypoints (List[Any]): A list (one per image) of keypoints (as NumPy arrays or lists).
        heights (tf.Tensor): Original heights of the images.
        widths (tf.Tensor): Original widths of the images.
        feature_dim (int): The dimensionality of the DINO features.

    Returns:
        List[tf.Tensor]: A list of descriptors for each image.
    """
    batch_size = tf.shape(dino_features)[0]

    heights_resized, widths_resized = _preprocess_shape(
        height, width, image_size_max=630, h_down_rate=14, w_down_rate=14
    )
    height_feat = heights_resized // 14
    width_feat = widths_resized // 14

    # Ensure width_feat and height_feat have a batch dimension
    if tf.rank(width_feat) == 0 or tf.shape(width_feat)[0] == 1:
        width_feat = tf.fill([batch_size], tf.cast(width_feat, tf.float32))
    if tf.rank(height_feat) == 0 or tf.shape(height_feat)[0] == 1:
        height_feat = tf.fill([batch_size], tf.cast(height_feat, tf.float32))

    # Get static values if possible
    height_feat_val = tf.get_static_value(height_feat)[0]
    width_feat_val = tf.get_static_value(width_feat)[0]
    new_shape = [batch_size, height_feat_val, width_feat_val, feature_dim]
    dino_features = tf.reshape(dino_features, new_shape)

    # Create image size and feature size tensors
    heights_tensor = tf.fill([batch_size], tf.cast(height, tf.float32))
    widths_tensor = tf.fill([batch_size], tf.cast(width, tf.float32))
    img_size = tf.cast(tf.stack([widths_tensor[0], heights_tensor[0]]), tf.float32)
    feature_size = tf.cast(tf.stack([width_feat_val, height_feat_val]), tf.float32)

    dino_descriptors = []

    # Loop through each image in the batch
    for i in range(batch_size):
        keypoints_tensor = tf.convert_to_tensor(keypoints[i], dtype=tf.float32)
        kp_scaled = keypoints_tensor / img_size * feature_size
        kp_scaled = tf.clip_by_value(kp_scaled, 0, feature_size - 1)

        descriptor_map = dino_features[i]
        descriptors_for_image = lookup_descriptor_bilinear(kp_scaled, descriptor_map)
        dino_descriptors.append(descriptors_for_image)

    return dino_descriptors

def lookup_descriptor_bilinear(keypoints, descriptor_map):
    """
    Looks up descriptors at given keypoints via bilinear interpolation on the descriptor map.

    Args:
        keypoints (tf.Tensor): A tensor of keypoints with shape (N, 2).
        descriptor_map (tf.Tensor): A tensor of descriptors with shape (height, width, depth).

    Returns:
        tf.Tensor: Interpolated descriptors with shape (N, depth).
    """
    keypoints = tf.cast(keypoints, tf.float32)
    descriptor_map = tf.cast(descriptor_map, tf.float32)

    height = tf.cast(tf.shape(descriptor_map)[0], tf.float32)
    width = tf.cast(tf.shape(descriptor_map)[1], tf.float32)

    x = keypoints[:, 0]
    y = keypoints[:, 1]

    # Compute floor and ceil coordinates
    x0 = tf.floor(x)
    x1 = x0 + 1
    y0 = tf.floor(y)
    y1 = y0 + 1

    # Clip coordinates to be within valid range
    x0 = tf.clip_by_value(x0, 0, width - 1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y0 = tf.clip_by_value(y0, 0, height - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)

    # Gather descriptor values at the four surrounding pixels
    Ia = tf.gather_nd(descriptor_map, tf.cast(tf.stack([y0, x0], axis=-1), tf.int32))
    Ib = tf.gather_nd(descriptor_map, tf.cast(tf.stack([y1, x0], axis=-1), tf.int32))
    Ic = tf.gather_nd(descriptor_map, tf.cast(tf.stack([y0, x1], axis=-1), tf.int32))
    Id = tf.gather_nd(descriptor_map, tf.cast(tf.stack([y1, x1], axis=-1), tf.int32))

    # Compute interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    wa = tf.cast(wa, tf.float32)
    wb = tf.cast(wb, tf.float32)
    wc = tf.cast(wc, tf.float32)
    wd = tf.cast(wd, tf.float32)

    # Compute the final descriptor via bilinear interpolation
    descriptor = wa[:, tf.newaxis] * Ia + wb[:, tf.newaxis] * Ib + wc[:, tf.newaxis] * Ic + wd[:, tf.newaxis] * Id

    return descriptor