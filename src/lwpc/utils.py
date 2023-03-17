import numpy as np


def se3inv(pose: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a SE(3) matrix `pose`. Expects shapes (4, 4) or (N, 4, 4).
    """
    # Check type.
    if not isinstance(pose, np.ndarray):
        raise TypeError(f"Expected an ND array, but received {type(pose)}.")

    # Check shape.
    if len(pose.shape) not in [2, 3] or pose.shape[-2:] != (4, 4):
        raise ValueError("Expected (4,4) or (N,4,4) array.")

    # Check last row(s).
    if not np.array_equiv(pose[..., 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("Expected the last row to be: [ 0., 0., 0., 1. ].")

    R_T = pose[..., :3, :3].T
    t = pose[..., :3, 3]

    rv = np.empty_like(pose)
    rv[..., :3, :3] = R_T
    rv[..., :3, 3] = np.squeeze(-R_T.reshape(-1, 3, 3) @ t.reshape(-1, 3, 1))
    rv[..., 3, :] = [0.0, 0.0, 0.0, 1.0]

    return rv


def project(
    intrinsics_matrix: np.ndarray,
    camera_pose: np.ndarray,
    points: np.ndarray,
    return_hip: bool,
) -> np.ndarray:
    """
    Given the camera intrinsics matrix `intrinsics_matrix`, the camera pose `camera_pose`,
    and the points `points`, project the points onto an image plane. If `return_hip` == True,
    then this function returns image point in homogeneous coordinates: [ x, y, w ]^T. Note:
    `points` should be in format (N, 3) for non-homogeneous coordinates or (N, 4) for homogeneous
    coordinates.
    """
    # Sanity checks.
    if not isinstance(intrinsics_matrix, np.ndarray):
        raise TypeError(
            f"Camera intrinsics matrix should be an ND array. Got type {type(intrinsics_matrix)}."
        )

    if type(camera_pose) is not np.ndarray:
        raise TypeError(
            f"Camera pose should be an ND array. Got type {type(camera_pose)}."
        )

    if type(points) is not np.ndarray:
        raise TypeError(f"Points should be an ND array. Got type {type(points)}.")

    if points.shape == (3,):
        points = points.reshape(1, 3)

    # Convert to homogenous coordinates if necessary.
    if len(points.shape) == 2 and points.shape[1] == 3:
        points = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))

    # Sanity check before projection.
    if len(points.shape) != 2 or points.shape[1] != 4 or len(points) == 0:
        raise ValueError("Points must have shape (N,4).")

    if not np.array_equal(points[:, 3], [1.0] * len(points)):
        raise NotImplementedError("Expected the last column to have all ones.")

    # Project the points onto the image frame and normalize.
    projected_points = intrinsics_matrix @ se3inv(camera_pose) @ points.transpose()
    if return_hip == False:
        projected_points = projected_points[:2] / projected_points[2]

    return np.transpose(projected_points)
