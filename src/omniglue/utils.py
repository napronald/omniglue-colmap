import os
import time
import logging
from functools import wraps
from typing import Callable, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from pycolmap import TwoViewGeometryConfiguration 

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def timed(func: Callable) -> Callable:
    """
    A decorator that logs the execution time of the function.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

def soft_assignment_to_match_matrix(
    soft_assignment: tf.Tensor, match_threshold: float
) -> tf.Tensor:
    """
    Converts a matrix of soft assignment values to a binary yes/no match matrix
    using mutual nearest neighbor checks and thresholding.

    :param soft_assignment: (B, N, M) tensor. Matching likelihood between sets
      of features (N in image0, M in image1).
    :param match_threshold: float, threshold to consider a match valid.
    :return: A boolean tensor of shape (B, N, M). A `True` indicates a match.
    """
    max0 = tf.reduce_max(soft_assignment, axis=2)   
    indices0 = tf.argmax(soft_assignment, axis=2)   
    indices1 = tf.argmax(soft_assignment, axis=1)  

    B = tf.shape(soft_assignment)[0]
    N = tf.shape(soft_assignment)[1]
    M = tf.shape(soft_assignment)[2]

    row_idx = tf.tile(tf.expand_dims(tf.range(N, dtype=indices1.dtype), 0), [B, 1]) 
    col_idx = indices0  
    row_gathered = tf.gather(indices1, col_idx, batch_dims=1)  
    mutual = tf.equal(row_gathered, row_idx)  

    match_mask = tf.cast(mutual, soft_assignment.dtype)     
    match_scores = match_mask * max0   

    oh = tf.one_hot(col_idx, depth=M, dtype=soft_assignment.dtype)   
    match_scores_expanded = tf.expand_dims(match_scores, axis=-1)   
    match_matrix_f = oh * match_scores_expanded                     
    match_matrix = match_matrix_f > match_threshold
    return match_matrix

def normalize_descriptors(desc: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(desc, axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    return desc / norm

def brute_force(desc0: np.ndarray, desc1: np.ndarray, ratio: float = 0.75) -> np.ndarray:
    """
    Standard brute-force matching for fallback if the deep matching yields too few matches.
    """
    desc0_norm = normalize_descriptors(desc0)
    desc1_norm = normalize_descriptors(desc1)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(desc0_norm, desc1_norm, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m.queryIdx, m.trainIdx])
    return np.array(good_matches, dtype=np.int32)

def construct_camera_intrinsics(camera_params: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the camera intrinsic matrix K and distortion coefficients D 
    from the provided camera parameters: fx, fy, cx, cy, k1, k2, p1, p2.
    """
    fx, fy, cx, cy, k1, k2, p1, p2 = camera_params
    K = np.array([
        [fx,  0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    D = np.array([k1, k2, p1, p2], dtype=np.float64)
    return K, D

def rotm_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a quaternion [w, x, y, z].
    """
    q = np.empty(4, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        t = np.sqrt(1.0 + t)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = 0 if R[1, 1] <= R[0, 0] else 1
        if R[2, 2] > R[i, i]:
            i = 2
        j, k = (i + 1) % 3, (i + 2) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j + 1] = (R[j, i] + R[i, j]) * t
        q[k + 1] = (R[k, i] + R[i, k]) * t
    return q

def compute_two_view_geometry(
    pts1: np.ndarray, 
    pts2: np.ndarray, 
    K: np.ndarray, 
    D: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Fundamental (F), Essential (E), Homography (H), 
    and the relative pose (q, t) between two sets of matched points 
    (pts1, pts2).
    """
    N1 = cv2.fisheye.undistortPoints(pts1, K, D)
    N2 = cv2.fisheye.undistortPoints(pts2, K, D)

    enough_matches = (N1 is not None and N2 is not None and len(N1) >= 4 and len(N2) >= 4)

    if not enough_matches:
        logger.info("[WARNING] Not enough matches => returning identity geometry.")
        return (
            np.eye(3),  
            np.eye(3),  
            np.eye(3), 
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),  
            np.zeros((3, 1), dtype=np.float64)  
        )

    F, _ = cv2.findFundamentalMat(N1, N2, cv2.FM_RANSAC)
    if F is None:
        F = np.eye(3)

    E, _ = cv2.findEssentialMat(N1, N2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        E = np.eye(3)

    ret, R, tvec, _ = cv2.recoverPose(E, N1, N2, K)
    if not ret:
        R = np.eye(3)
        tvec = np.zeros((3,1))

    H, _ = cv2.findHomography(N1, N2, cv2.RANSAC)
    if H is None:
        H = np.eye(3)

    qvec = rotm_to_quaternion(R)
    return F, E, H, qvec, tvec

def determine_geometry_config(F: np.ndarray, E: np.ndarray, H: np.ndarray) -> TwoViewGeometryConfiguration:
    """
    Determines the appropriate TwoViewGeometryConfiguration based on 
    the estimated F, E, H matrices using rank checks and non-zero checks.
    """
    EPS = 1e-6

    if E is not None and not np.allclose(E, 0, atol=EPS):
        rank_E = np.linalg.matrix_rank(E)
        if rank_E == 2:
            return TwoViewGeometryConfiguration.CALIBRATED
        else:
            return TwoViewGeometryConfiguration.DEGENERATE

    if F is not None and not np.allclose(F, 0, atol=EPS):
        rank_F = np.linalg.matrix_rank(F)
        if rank_F == 2:
            return TwoViewGeometryConfiguration.UNCALIBRATED
        else:
            return TwoViewGeometryConfiguration.DEGENERATE

    if H is not None and not np.allclose(H, 0, atol=EPS):
        return TwoViewGeometryConfiguration.PLANAR_OR_PANORAMIC

    return TwoViewGeometryConfiguration.UNDEFINED

def get_index_by_id(colmap_db, image_id: int, image_paths: List[str]):
    """
    Returns the index in `image_paths` that corresponds to the given `image_id`.
    """
    name = colmap_db.get_image_name(image_id)
    for idx, path in enumerate(image_paths):
        if os.path.basename(path) == name:
            return idx
    return None

def visualize_matches(ref_img_path: str, succ_img_path: str, pts1: np.ndarray, pts2: np.ndarray, output_dir: str = "viz"):
    """
    Visualize feature matches.
    """
    ref = np.array(Image.open(ref_img_path).convert("RGB"))
    succ = np.array(Image.open(succ_img_path).convert("RGB"))

    viz = np.hstack((ref, succ))
    w0 = ref.shape[1]

    # Draw lines between matching keypoints
    for pt1, pt2 in zip(pts1, pts2):
        pt1 = tuple(np.array(pt1).flatten().astype(np.int32))
        pt2 = tuple((np.array(pt2).flatten() + np.array([w0, 0])).astype(np.int32))
        color = (0, 255, 0)
        cv2.line(viz, pt1, pt2, color, 2)

    # Draw circles on keypoints from the reference image
    for kp in pts1:
        kp_tuple = tuple(np.array(kp).flatten().astype(np.int32))
        cv2.circle(viz, kp_tuple, 4, (0, 0, 255), 2)

    # Draw circles on keypoints from the successor image
    for kp in pts2:
        kp = np.array(kp).flatten()
        kp[0] += w0  
        kp_tuple = tuple(kp.astype(np.int32))
        cv2.circle(viz, kp_tuple, 4, (0, 0, 255), 2)

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"{os.path.basename(ref_img_path)}_{os.path.basename(succ_img_path)}.png"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, viz)