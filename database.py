import sqlite3
import numpy as np
from typing import List, Any

MAX_IMAGE_ID = 2**31 - 1

def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    """
    Creates a unique pair_id from two image IDs, ensuring ordering 
    to avoid collisions.
    """
    return image_id1 * MAX_IMAGE_ID + image_id2 if image_id1 < image_id2 else image_id2 * MAX_IMAGE_ID + image_id1

def array_to_blob(array: np.ndarray) -> bytes:
    """
    Converts a NumPy array to a byte blob for database storage.
    """
    return array.tobytes()

CREATE_CAMERAS_TABLE = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL
)
"""

CREATE_IMAGES_TABLE = f"""
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
)
"""

CREATE_KEYPOINTS_TABLE = """
CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
)
"""

CREATE_DESCRIPTORS_TABLE = """
CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
)
"""

CREATE_MATCHES_TABLE = """
CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB
)
"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB
)
"""

CREATE_ALL_TABLES = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE
])

class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path: str) -> 'COLMAPDatabase':
        return sqlite3.connect(database_path, factory=COLMAPDatabase)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_tables()
    
    def create_tables(self):
        self.executescript(CREATE_ALL_TABLES)
    
    def add_camera(self, model: int, width: int, height: int, params: np.ndarray, 
                   prior_focal_length: bool = False) -> int:
        params_blob = array_to_blob(np.asarray(params, dtype=np.float64))
        cursor = self.execute(
            """
            INSERT INTO cameras (model, width, height, params, prior_focal_length) 
            VALUES (?, ?, ?, ?, ?)
            """,
            (model, width, height, params_blob, int(prior_focal_length))
        )
        return cursor.lastrowid
    
    def add_image(self, name: str, camera_id: int,
                  prior_qw: float = None, prior_qx: float = None, 
                  prior_qy: float = None, prior_qz: float = None, 
                  prior_tx: float = None, prior_ty: float = None, 
                  prior_tz: float = None) -> int:
        cursor = self.execute(
            """
            INSERT INTO images 
            (name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz)
        )
        return cursor.lastrowid
    
    def add_keypoints(self, image_id: int, keypoints: np.ndarray):
        rows, cols = keypoints.shape
        data_blob = array_to_blob(keypoints.astype(np.float32))
        cursor = self.execute("SELECT image_id FROM keypoints WHERE image_id = ?", (image_id,))
        if cursor.fetchone():
            self.execute(
                """
                UPDATE keypoints 
                SET rows = ?, cols = ?, data = ? 
                WHERE image_id = ?
                """,
                (rows, cols, data_blob, image_id)
            )
        else:
            self.execute(
                """
                INSERT INTO keypoints (image_id, rows, cols, data) 
                VALUES (?, ?, ?, ?)
                """,
                (image_id, rows, cols, data_blob)
            )
    
    def add_descriptors(self, image_id: int, descriptors: np.ndarray):
        rows, cols = descriptors.shape
        data_blob = array_to_blob(descriptors.astype(np.float32))
        cursor = self.execute("SELECT image_id FROM descriptors WHERE image_id = ?", (image_id,))
        if cursor.fetchone():
            self.execute(
                """
                UPDATE descriptors 
                SET rows = ?, cols = ?, data = ? 
                WHERE image_id = ?
                """,
                (rows, cols, data_blob, image_id)
            )
        else:
            self.execute(
                """
                INSERT INTO descriptors (image_id, rows, cols, data) 
                VALUES (?, ?, ?, ?)
                """,
                (image_id, rows, cols, data_blob)
            )
    
    def add_matches(self, image_id1: int, image_id2: int, matches: np.ndarray):
        rows, cols = matches.shape
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        data_blob = array_to_blob(matches.astype(np.int32))
        self.execute(
            """
            INSERT INTO matches (pair_id, rows, cols, data) 
            VALUES (?, ?, ?, ?)
            """,
            (pair_id, rows, cols, data_blob)
        )
    
    def add_two_view_geometry(self, 
                              image_id1: int, 
                              image_id2: int, 
                              matches: np.ndarray, 
                              F: np.ndarray = None, 
                              E: np.ndarray = None, 
                              H: np.ndarray = None, 
                              qvec: np.ndarray = None, 
                              tvec: np.ndarray = None, 
                              config: Any = 2):
        """
        Adds two-view geometry with the given configuration. 
        'config' can be an integer or an enum (e.g., TwoViewGeometryConfiguration).
        """
        if hasattr(config, 'value'):  
            config_val = config.value
        else:
            config_val = int(config)

        pair_id = image_ids_to_pair_id(image_id1, image_id2)

        F = np.eye(3, dtype=np.float64) if F is None else F
        E = np.eye(3, dtype=np.float64) if E is None else E
        H = np.eye(3, dtype=np.float64) if H is None else H
        qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if qvec is None else qvec
        tvec = np.zeros(3, dtype=np.float64) if tvec is None else tvec

        matches = np.asarray(matches, dtype=np.int32)
        rows, cols = matches.shape
        matches_blob = array_to_blob(matches)
        F_blob = array_to_blob(F.astype(np.float64))
        E_blob = array_to_blob(E.astype(np.float64))
        H_blob = array_to_blob(H.astype(np.float64))
        qvec_blob = array_to_blob(qvec.astype(np.float64))
        tvec_blob = array_to_blob(tvec.astype(np.float64))

        self.execute(
            """
            INSERT INTO two_view_geometries 
            (pair_id, rows, cols, data, config, F, E, H, qvec, tvec) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (pair_id, rows, cols, matches_blob, config_val, F_blob, E_blob, H_blob, qvec_blob, tvec_blob)
        )
    
    def get_all_images(self) -> List[tuple]:
        cursor = self.execute("SELECT image_id, name FROM images ORDER BY image_id ASC")
        return cursor.fetchall()
    
    def get_all_image_ids(self) -> List[int]:
        """
        Retrieve all image IDs from the images table, ordered by image_id ascending.
        """
        cursor = self.execute("SELECT image_id FROM images ORDER BY image_id ASC")
        rows = cursor.fetchall()
        image_ids = [row[0] for row in rows]
        return image_ids
    
    def get_image_name(self, image_id: int) -> str:
        """
        Retrieve the image name (file path) for a given image_id.
        """
        cursor = self.execute("SELECT name FROM images WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        return None
    
    def get_keypoints(self, image_id: int) -> np.ndarray:
        cursor = self.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            return np.zeros((0, 2), dtype=np.float32)
        rows, cols, data = row
        arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        return arr
    
    def get_descriptors(self, image_id: int) -> np.ndarray:
        cursor = self.execute("SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            return np.zeros((0, 256), dtype=np.float32)
        rows, cols, data = row
        arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        return arr


CREATE_SCORES_TABLE = """
CREATE TABLE IF NOT EXISTS scores (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
)
"""

CREATE_DESCRIPTORS_DINO_TABLE = """
CREATE TABLE IF NOT EXISTS descriptors_dino (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
)
"""

CREATE_FEATURE_DB_TABLES = "; ".join([
    CREATE_SCORES_TABLE,
    CREATE_DESCRIPTORS_DINO_TABLE,
])

class FeatureDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path: str) -> 'FeatureDatabase':
        return sqlite3.connect(database_path, factory=FeatureDatabase)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_tables()

    def create_tables(self):
        self.executescript(CREATE_FEATURE_DB_TABLES)

    def add_scores(self, image_id: int, scores: np.ndarray):
        """
        Store SuperPoint scores for a given image ID.
        """
        if scores.ndim == 1:
            scores = scores[:, None]

        rows, cols = scores.shape
        data_blob = array_to_blob(scores.astype(np.float32))

        cursor = self.execute("SELECT image_id FROM scores WHERE image_id = ?", (image_id,))
        if cursor.fetchone():
            self.execute(
                """
                UPDATE scores
                SET rows = ?, cols = ?, data = ?
                WHERE image_id = ?
                """,
                (rows, cols, data_blob, image_id)
            )
        else:
            self.execute(
                """
                INSERT INTO scores (image_id, rows, cols, data) 
                VALUES (?, ?, ?, ?)
                """,
                (image_id, rows, cols, data_blob)
            )

    def add_descriptors_dino(self, image_id: int, descriptors_dino: np.ndarray):
        """
        Store DINO descriptors for a given image ID.
        descriptors_dino is expected to be shape (N, 768).
        """
        rows, cols = descriptors_dino.shape
        data_blob = array_to_blob(descriptors_dino.astype(np.float32))

        cursor = self.execute("SELECT image_id FROM descriptors_dino WHERE image_id = ?", (image_id,))
        if cursor.fetchone():
            self.execute(
                """
                UPDATE descriptors_dino
                SET rows = ?, cols = ?, data = ?
                WHERE image_id = ?
                """,
                (rows, cols, data_blob, image_id)
            )
        else:
            self.execute(
                """
                INSERT INTO descriptors_dino (image_id, rows, cols, data) 
                VALUES (?, ?, ?, ?)
                """,
                (image_id, rows, cols, data_blob)
            )

    def get_scores(self, image_id: int) -> np.ndarray:
        cursor = self.execute("SELECT rows, cols, data FROM scores WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            return np.zeros((0, 1), dtype=np.float32)
        rows, cols, data = row
        arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        return arr

    def get_descriptors_dino(self, image_id: int) -> np.ndarray:
        cursor = self.execute("SELECT rows, cols, data FROM descriptors_dino WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            return np.zeros((0, 768), dtype=np.float32)
        rows, cols, data = row
        arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        return arr