# Adapted from https://github.com/google-research/omniglue/blob/main/src/omniglue/superpoint_extract.py
# with added batch processing support, mask filtering, and omniglue input compatibility.

import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """Removes keypoints too close to the border."""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    """Keeps only the top k keypoints based on score."""
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores.squeeze(dim=1), k, dim=0)
    return keypoints[indices], scores.unsqueeze(1)


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations."""
    b, c, h, w = descriptors.shape
    # Convert from (x, y) to the normalized coordinates that grid_sample expects.
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)

    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def filter_with_mask(scores, masks, ignored_values={5, 7, 11}):
    """
    Zero out scores at locations where the mask value is in ignored_values.
    """
    valid_mask = torch.ones_like(masks, dtype=torch.bool)
    for val in ignored_values:
        valid_mask &= (masks != val)
    return scores * valid_mask.float()


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        # Detector layers
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor layers
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0
        )

        # Load pretrained weights
        path = (
            "https://github.com/magicleap/"
            "SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth"
        )
        state_dict = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        self.load_state_dict(state_dict)

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or "-1"')

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        ignored_values={0}
    ):
        """
        Forward pass of the SuperPoint model with segmentation-based filtering.

        Args:
            images: (B, 1, H, W) input images.
            masks: (B, H, W) segmentation masks (same spatial dims as images).
            ignored_values: Set of mask values to ignore during keypoint filtering.

        Returns:
            keypoints_np: list of (N, 2) arrays with (x, y) keypoints.
            descriptors_np: list of (N, descriptor_dim) arrays.
            scores_np: list of (N, 1) arrays with detection scores.
        """
        # --- Feature Extraction ---
        x = self.relu(self.conv1a(images))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # --- Keypoint Scoring ---
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        # Convert [B,65,H',W'] -> [B,H',W',65], then split off the last channel
        scores = torch.nn.functional.softmax(scores, dim=1)[:, :-1]
        b, _, h, w = scores.shape

        # Reshape to full-resolution: (B, H', W', 8, 8) -> (B, H'*8, W'*8)
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

        # Zero-out scores in ignored areas
        scores = filter_with_mask(scores, masks, ignored_values=ignored_values)

        # Apply non-maximum suppression
        scores = simple_nms(scores, self.config['nms_radius'])

        # --- Keypoint Extraction ---
        net_keypoints = [torch.nonzero(s > self.config['keypoint_threshold']) for s in scores]
        net_scores_list = [s[tuple(k.t())] for s, k in zip(scores, net_keypoints)]

        kpts_filtered, scores_filtered = [], []
        img_height, img_width = images.shape[2], images.shape[3]

        for batch_index, (kpts, score_tensor) in enumerate(zip(net_keypoints, net_scores_list)):
            # Already thresholded by net_keypoints
            filtered_keypoints = [[j, i] for (i, j) in kpts]
            filtered_scores = [sc.item() for sc in score_tensor]

            kpts_tensor = torch.tensor(filtered_keypoints, dtype=torch.float32, device='cpu')
            scores_tensor = torch.tensor(filtered_scores, dtype=torch.float32, device='cpu').unsqueeze(1)

            # Remove keypoints close to the image borders
            kpts_tensor, scores_tensor = remove_borders(
                kpts_tensor, scores_tensor,
                self.config['remove_borders'],
                img_height, img_width
            )

            # Retain only the top-K keypoints (if configured)
            if self.config['max_keypoints'] > 0:
                kpts_tensor, scores_tensor = top_k_keypoints(
                    kpts_tensor, scores_tensor, self.config['max_keypoints']
                )

            kpts_filtered.append(kpts_tensor)
            scores_filtered.append(scores_tensor)

        # --- Descriptor Extraction ---
        descriptors_map = self.relu(self.convDa(x))
        descriptors_map = self.convDb(descriptors_map)
        descriptors_map = torch.nn.functional.normalize(descriptors_map, p=2, dim=1)

        descriptors_sampled = []
        for kpts, d_map in zip(kpts_filtered, descriptors_map):
            kpts_device = kpts.to(d_map.device)
            sampled_desc = sample_descriptors(kpts_device[None], d_map[None], s=8)[0].permute(1, 0)
            descriptors_sampled.append(sampled_desc)

        # Convert to numpy
        keypoints_np = [k.cpu().numpy() for k in kpts_filtered]
        descriptors_np = [d.cpu().numpy() for d in descriptors_sampled]
        scores_np = [s.cpu().numpy() for s in scores_filtered]

        return keypoints_np, descriptors_np, scores_np