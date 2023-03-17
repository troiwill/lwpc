from lwpc import utils
import numpy as np
from typing import Optional


class PerspectiveCameraModel:
    def __init__(self, intrins_mat: Optional[np.ndarray] = None) -> None:
        self.__intrins_mat = None
        if intrins_mat is not None:
            self.set_intrinsics_matrix(intrins_mat)

    @property
    def intrinsics_matrix(self) -> np.ndarray:
        """
        Returns the camera intristrics matrix.
        """
        return self.__C.copy()

    def set_intrinsics_matrix(self, intrins_mat: np.ndarray) -> None:
        """
        Sets the camera intrinsics matrix `intrins_mat`. The camera matrix should have the shape (3,3) or (3,4).
        """
        # Sanity check.
        if type(intrins_mat) is not np.ndarray:
            raise TypeError(
                f"Camera instrinsics matrix `intrins_mat` should be an NumPy array. Got type {type(intrins_mat)}."
            )

        if intrins_mat.shape == (3, 3):
            self.__C = np.hstack(
                (intrins_mat, np.zeros_like(intrins_mat, shape=(3, 1)))
            )

        elif intrins_mat.shape == (3, 4):
            if not np.array_equal(intrins_mat[:, 3].flatten(), [0.0, 0.0, 0.0]):
                raise ValueError(f"Expected the last column to have zeros.")
            self.__intrins_mat = intrins_mat.copy()

        else:
            raise ValueError(
                f"Expected matrix with shape (3,3) or (3,4). Got shape = {intrins_mat.shape}."
            )

    def project(
        self, camera_pose: np.ndarray, points: np.ndarray, return_hip: bool = False
    ) -> np.ndarray:
        """
        Projects the `points` from the world frame to the image frame using the camera intrinsics matrix
        and the camera pose `camera_pose`.
        """
        # Sanity check.
        assert self.__intrins_mat is not None

        # Perform the projection.
        return utils.project(
            intrinsics_matrix=self.__intrins_mat,
            camera_pose=camera_pose,
            points=points,
            return_hip=return_hip,
        )
