import logging
from typing import List

import numpy as np
import tensorflow as tf
from omniglue import utils

logger = logging.getLogger(__name__)

class OmniGlue:
    def __init__(self, og_export: str, camera: List[float] = None, memory_limit: int = 2560, num_gpus: int = 1):
        """
        Initializes the OmniGlue object for feature matching. 
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)] * num_gpus
            )
        self.matcher = tf.saved_model.load(og_export)
        self.K, self.D = utils.construct_camera_intrinsics(camera)

    def load_features(
        self,
        colmap_db,
        feature_db,
        image_id: int
    ):
        """
        Loads an image and its corresponding features from the databases.
        """
        kpts = colmap_db.get_keypoints(image_id)
        desc = colmap_db.get_descriptors(image_id)
        scores = feature_db.get_scores(image_id)
        dino_desc = feature_db.get_descriptors_dino(image_id)
        return kpts, desc, scores, dino_desc

    def find_matches(
        self,
        colmap_db,
        feature_db,
        image_paths: List[str],
        args: List[int],
        min_matches: int = 8
    ):
        """
        Main function to perform sequential feature matches in batches. 
        """
        image_ids = colmap_db.get_all_image_ids()
        total_images = len(image_ids)
        logger.info(f"Starting match finding for {total_images} images.")

        for i, ref_id in enumerate(image_ids[:-1]):
            ref_kpts, ref_desc, ref_scores, ref_dino_desc = self.load_features(colmap_db, feature_db, ref_id)

            successors_end = min(i + 1 + args.max_successors, total_images)
            successor_ids = image_ids[i + 1:successors_end]
            num_successors = len(successor_ids)

            for batch_start in range(0, num_successors, args.batch_size_match):
                batch_ids = successor_ids[batch_start:batch_start + args.batch_size_match]
                for succ_id in batch_ids:
                    succ_kpts, succ_desc, succ_scores, succ_dino_desc = self.load_features(colmap_db, feature_db, succ_id)

                    inputs = {
                        'keypoints0': tf.constant(ref_kpts[np.newaxis], dtype=tf.float32), 
                        'keypoints1': tf.constant(succ_kpts[np.newaxis], dtype=tf.float32),  
                        'descriptors0': tf.constant(ref_desc[np.newaxis], dtype=tf.float32),  
                        'descriptors1': tf.constant(succ_desc[np.newaxis], dtype=tf.float32),
                        'scores0': tf.constant(ref_scores.squeeze()[np.newaxis, :, np.newaxis], dtype=tf.float32),
                        'scores1': tf.constant(succ_scores.squeeze()[np.newaxis, :, np.newaxis], dtype=tf.float32),
                        'descriptors0_dino': tf.constant(ref_dino_desc[np.newaxis], dtype=tf.float32),
                        'descriptors1_dino': tf.constant(succ_dino_desc[np.newaxis], dtype=tf.float32),
                        'width0': tf.constant([args.camera_width], dtype=tf.int32),
                        'height0': tf.constant([args.camera_height], dtype=tf.int32),
                        'width1': tf.constant([args.camera_width], dtype=tf.int32),
                        'height1': tf.constant([args.camera_height], dtype=tf.int32),
                    }

                    match_indices = np.empty((0, 2), dtype=np.uint32)
                    MATCH_THRESHOLD = 1e-3

                    while match_indices.shape[0] < min_matches and MATCH_THRESHOLD > 1e-5:
                        og_out = self.matcher.signatures['serving_default'](**inputs)

                        soft_assignment = og_out['soft_assignment'][:, :-1, :-1]  
                        match_matrix = utils.soft_assignment_to_match_matrix(soft_assignment, MATCH_THRESHOLD)
                        match_matrix = match_matrix.numpy().squeeze()  

                        match_indices = np.argwhere(match_matrix > 0).astype(np.uint32)
                        if match_indices.shape[0] < min_matches:
                            MATCH_THRESHOLD *= 0.1

                    if match_indices.shape[0] < min_matches:
                        logger.info(
                            f"Not enough deep matches between Image {ref_id} and {succ_id}. Using brute-force fallback."
                        )
                        bf_matches = np.empty((0, 2), dtype=np.uint32)
                        threshold = 0.75
                        while bf_matches.shape[0] < min_matches and threshold < 1.0:
                            bf_matches = utils.brute_force(desc0=ref_desc, desc1=succ_desc, ratio=threshold)
                            threshold += 0.05

                        all_matches = np.concatenate((match_indices, bf_matches), axis=0)
                    else:
                        all_matches = match_indices

                    if all_matches.shape[0] > 0:
                        all_matches = np.unique(all_matches, axis=0).astype(np.uint32)
                    else:
                        logger.info(f"No matches found between Image {ref_id} and Image {succ_id}. Skipping geometry.")
                        continue

                    pts1 = ref_kpts[all_matches[:, 0]].astype(np.float32).reshape(-1, 1, 2)
                    pts2 = succ_kpts[all_matches[:, 1]].astype(np.float32).reshape(-1, 1, 2)

                    F, E, H, q, t = utils.compute_two_view_geometry(pts1, pts2, self.K, self.D)
                    config = utils.determine_geometry_config(F, E, H)

                    colmap_db.add_matches(ref_id, succ_id, all_matches)
                    colmap_db.add_two_view_geometry(ref_id, succ_id, all_matches, F, E, H, q, t, config)

                    if args.visualize:
                        ref_idx = utils.get_index_by_id(colmap_db, ref_id, image_paths)
                        succ_idx = utils.get_index_by_id(colmap_db, succ_id, image_paths)
                        ref_image_path = image_paths[ref_idx]
                        succ_image_path = image_paths[succ_idx]
                        utils.visualize_matches(ref_image_path, succ_image_path, pts1, pts2)

                colmap_db.commit()
                logger.info(f"Matched Image {i+1}/{total_images}.")