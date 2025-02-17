import os
import logging
import argparse
from typing import Tuple, List

import cv2
import torch
import numpy as np
from PIL import Image

from omniglue import dino_extract, superpoint_extract, omniglue_extract
from database import COLMAPDatabase, FeatureDatabase  
from omniglue.utils import timed

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

@timed
def load_images_and_masks(image_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
    image_paths = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png')))
    mask_paths = [os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + '.png') for f in image_paths]
    logger.info(f"Loaded {len(image_paths)} images and masks.")
    return image_paths, mask_paths

@timed
def extract_superpoint_features(
    colmap_db,
    image_paths: List[str],
    mask_paths: List[str],
    sp_model: superpoint_extract.SuperPoint,
    device: str,
    batch_size_sp: int,
    resize: Tuple[int, int] = (640, 480)
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]], List[np.ndarray]]:
    """
    Extracts SuperPoint features for images in batches, with optional resizing prior to feature extraction.
    """
    sp_results = []
    all_imgs_rgb = []
    total_images = len(image_paths)
    num_batches = (total_images + batch_size_sp - 1) // batch_size_sp

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size_sp
        end = min(start + batch_size_sp, total_images)
        batch_image_paths = image_paths[start:end]
        batch_mask_paths = mask_paths[start:end]

        batch_gray_list = []
        batch_masks_list = []
        batch_image_ids = []
        scales = []  

        for img_path, msk_path in zip(batch_image_paths, batch_mask_paths):
            image_id = colmap_db.add_image(name=os.path.basename(img_path), camera_id=1)
            batch_image_ids.append(image_id)

            original_img = np.array(Image.open(img_path).convert("RGB"))
            all_imgs_rgb.append(original_img)

            if resize is not None:
                processed_img = cv2.resize(original_img, resize, interpolation=cv2.INTER_LINEAR)
                scale_x = original_img.shape[1] / float(resize[0])
                scale_y = original_img.shape[0] / float(resize[1])
                scales.append((scale_x, scale_y))
            else:
                processed_img = original_img
                scales.append((1.0, 1.0))

            gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            batch_gray_list.append(gray[None, ...])

            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if resize is not None:
                mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)
            batch_masks_list.append(mask)

        batch_imgs_gray = torch.from_numpy(np.stack(batch_gray_list, axis=0))
        batch_masks = torch.from_numpy(np.stack(batch_masks_list, axis=0))

        sp_model.to(device).eval()
        with torch.no_grad():
            kpts_list, desc_list, scores_list = sp_model(batch_imgs_gray.to(device), batch_masks.to(device))

        for i, image_id in enumerate(batch_image_ids):
            keypoints = kpts_list[i]
            scale = scales[i]
            scaled_kpts = keypoints.copy()
            scaled_kpts[:, 0] *= scale[0]
            scaled_kpts[:, 1] *= scale[1]
            sp_results.append((scaled_kpts, desc_list[i], scores_list[i], image_id))

        colmap_db.commit()
        logger.info(f"Processed SuperPoint batch {batch_idx + 1}/{num_batches}.")

    return sp_results, all_imgs_rgb

@timed
def extract_dino_features(
    all_imgs_rgb: List[np.ndarray],
    kpts_list: List[np.ndarray],
    dino_model: dino_extract.DINOExtract,
    batch_size_dino: int
) -> List[np.ndarray]:
    """
    Extracts DINO features in batches and computes DINO descriptors based on keypoints.
    """
    dino_desc_list = []
    total_images = len(all_imgs_rgb)
    num_batches = (total_images + batch_size_dino - 1) // batch_size_dino

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size_dino
        end = min(start + batch_size_dino, total_images)
        batch_rgb = all_imgs_rgb[start:end]
        height, width = batch_rgb[0].shape[:2]

        dino_features = dino_model(batch_rgb)
        batch_kpts = kpts_list[start:end]
        batch_dino_desc = dino_extract.get_dino_descriptors(
            dino_features, batch_kpts, height, width, 768
        )

        batch_dino_desc = [desc.numpy() for desc in batch_dino_desc]
        dino_desc_list.extend(batch_dino_desc)
        logger.info(f"Processed DINO batch {batch_idx + 1}/{num_batches}.")

    return dino_desc_list

def main():
    parser = argparse.ArgumentParser(description="OmniGlue Matching with colmap compatibility.")
    parser.add_argument('--image_dir', default='/path/to/images', help='Directory containing images.')
    parser.add_argument('--mask_dir', default='/path/to/masks', help='Directory containing segmentation masks.')
    parser.add_argument('--colmap_database_path', type=str, default='colmap_database.db', help='Path to the COLMAP database file.')
    parser.add_argument('--feature_database_path', type=str, default='feature_database.db', help='Path to the Feature database file.')
    parser.add_argument('--max_successors', type=int, default=10, help='Maximum number of subsequent images to match with each image.')

    # Batch sizes for feature extraction.
    parser.add_argument('--batch_size_sp', type=int, default=32, help='Batch size for SuperPoint feature extraction.')
    parser.add_argument('--batch_size_dino', type=int, default=32, help='Batch size for DINO feature extraction.')
    parser.add_argument('--batch_size_match', type=int, default=64, help='Batch size for feature matching.')

    # Visualization and extraction flags.
    parser.add_argument('--visualize', action='store_true', help='Visualizes matches using OpenCV.')
    parser.add_argument('--extract', action='store_true', help='Extracts features (SuperPoint and DINO) and stores them in the databases.')

    # Replace with Image Resolution
    parser.add_argument('--camera_width', type=int, default=1920, help='Camera image width.')
    parser.add_argument('--camera_height', type=int, default=1536, help='Camera image height.')

    # For colmap export, example values for a fisheye camera model
    parser.add_argument('--camera_model', type=int, default=5, help='Camera model ID.')
    parser.add_argument('--prior_focal_length', type=int, default=1, help='Prior focal length flag (1 for fixed, 0 otherwise).')
    parser.add_argument('--camera_params', type=float, nargs=8,
                        default=[489.167147, 489.581538, 960.420367, 770.178498, 0.070316, -0.011815, -0.000502, 0.000864],
                        help='Camera parameters: fx, fy, cx, cy, k1, k2, p1, p2.')
    args = parser.parse_args()

    #TODO add support for image only input

    colmap_db = COLMAPDatabase.connect(args.colmap_database_path)
    feature_db = FeatureDatabase.connect(args.feature_database_path)

    colmap_db.create_tables()
    feature_db.create_tables()

    colmap_db.add_camera(args.camera_model, args.camera_width, args.camera_height, np.array(args.camera_params), args.prior_focal_length)
    image_paths, mask_paths = load_images_and_masks(args.image_dir, args.mask_dir)

    og = omniglue_extract.OmniGlue(og_export="./models/og_export", camera=args.camera_params)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.extract:
        sp_model = superpoint_extract.SuperPoint({
            'nms_radius': 4, 'keypoint_threshold': 0.015, 'max_keypoints': 1024
        }).to(device).eval()
        
        # Extract SuperPoint features.
        sp_results, all_imgs_rgb = extract_superpoint_features(colmap_db, image_paths, mask_paths, sp_model, device, args.batch_size_sp)
        kpts_list = [res[0] for res in sp_results]
        
        # Extract DinoV2 features.
        dino_model = dino_extract.DINOExtract("./models/dinov2_vitb14_pretrain.pth", feature_layer=1)

        dino_desc_list = extract_dino_features(all_imgs_rgb, kpts_list, dino_model, args.batch_size_dino)
        
        # Store the features in the respective databases.
        for res, dino_desc in zip(sp_results, dino_desc_list):
            kpts, desc, scores, image_id = res
            colmap_db.add_keypoints(image_id, kpts.astype(np.float32))
            colmap_db.add_descriptors(image_id, desc.astype(np.float32))
            feature_db.add_scores(image_id, scores.astype(np.float32))
            feature_db.add_descriptors_dino(image_id, dino_desc.astype(np.float32))

        colmap_db.commit()
        feature_db.commit()

    # Perform Sequential Feature Matching
    og.find_matches(colmap_db, feature_db, image_paths, args)

    colmap_db.close()
    feature_db.close()


if __name__ == "__main__":
    main()