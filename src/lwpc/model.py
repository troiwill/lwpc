from lwpc import utils
import numpy as np
from typing import Optional


class PerspectiveCameraModel:
    def __init__(self, C: Optional[np.ndarray] = None) -> None:
        self.__C = None
        if C is not None:
            self.set_camera_matrix(C)

    @property
    def C(self) -> np.ndarray:
        """
        Returns the camera (intristrics) matrix.
        """
        return self.__C.copy()

    def set_camera_matrix(self, C: np.ndarray) -> None:
        """
        Sets the camera matrix `C`. The camera matrix should have the shape (3,3) or (3,4).
        """
        # Sanity check.
        if type(C) is not np.ndarray:
            raise TypeError(
                f"Camera matrix `C` should be an NumPy array. Got type {type(C)}."
            )

        if C.shape == (3, 3):
            self.__C = np.hstack((C, np.zeros_like(C, shape=(3, 1))))

        elif C.shape == (3, 4):
            if not np.array_equal(C[:, 3].flatten(), [0.0, 0.0, 0.0]):
                raise ValueError(f"Expected the last column to have zeros.")
            self.__C = C.copy()

        else:
            raise ValueError(
                f"Expected matrix with shape (3,3) or (3,4). Got shape = {C.shape}."
            )

    def project(self, camera_pose: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Projects the `points` from the world frame to the image frame using the camera matrix
        and the camera pose `camera_pose`.
        """
        return utils.project(
            camera_matrix=self.__C, camera_pose=camera_pose, points=points
        )
