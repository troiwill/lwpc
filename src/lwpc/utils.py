import numpy as np


def prepare_points_for_projection(points: np.ndarray, D: int) -> np.ndarray:
    """
    Performs a set of operations and sanity checks to prepare the points `points` for a project_* function. `D` is the dimension of a point (in non-homogeneous coordinates).
    """
    if type(points) is not np.ndarray:
        raise TypeError(f"Points should be an ND array. Got type {type(points)}.")

    if points.shape == (D,):
        points = points.reshape(1, D)

    # Convert to homogenous coordinates if necessary.
    if len(points.shape) == 2 and points.shape[1] == D:
        points = to_homogeneous_coords(points=points)

    # Sanity check to ensure the appropriate shape.
    if len(points.shape) != 2 or points.shape[1] != D + 1 or len(points) == 0:
        raise ValueError(f"Points must have shape (N,{D + 1}).")

    # Sanity check to ensure the last column has ones.
    if not np.array_equal(points[:, D], [1.0] * len(points)):
        raise NotImplementedError("Expected the last column to have all ones.")

    return points


def project_world_to_image(
    intrinsics_matrix: np.ndarray,
    camera_pose: np.ndarray,
    points: np.ndarray,
    return_nonhc: bool,
) -> np.ndarray:
    """
    Given the camera intrinsics matrix `intrinsics_matrix`, the camera pose `camera_pose`,
    and the points `points`, project the points onto an image plane. If `return_nonhc` == True,
    then this function returns image point in non-homogeneous coordinates: [ x, y ]^T. Note:
    `points` should be in format (N, 3) for non-homogeneous coordinates or (N, 4) for homogeneous
    coordinates.
    """
    # Sanity checks.
    if not isinstance(intrinsics_matrix, np.ndarray):
        raise TypeError(
            f"Camera intrinsics matrix should be an ND array. Got type {type(intrinsics_matrix)}."
        )

    if intrinsics_matrix.shape == (3, 3):
        intrinsics_matrix = np.hstack((intrinsics_matrix, np.zeros((3, 1))))

    if intrinsics_matrix.shape != (3, 4):
        raise ValueError(
            f"Expected an intrinsics matrix with shape (3, 3) or (3, 4), but got {intrinsics_matrix.shape}."
        )

    if type(camera_pose) is not np.ndarray:
        raise TypeError(
            f"Camera pose should be an ND array. Got type {type(camera_pose)}."
        )

    # Prepare the points for projection.
    points = prepare_points_for_projection(points=points, D=3)

    # Project the points onto the image frame.
    projected_points = intrinsics_matrix @ se3inv(camera_pose) @ points.transpose()
    if return_nonhc == True:
        return to_nonhomogeneous_coords(projected_points.T)

    else:
        return np.transpose(projected_points)


def project_image_to_world(
    intrinsics_matrix: np.ndarray,
    camera_pose: np.ndarray,
    points: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """
    Given the camera intrinsics matrix `intrinsics_matrix`, the camera pose `camera_pose`, and a set of points in the image space `points`,
    produce a ray that goes from the camera's coordinate frame through the image points and into the world. With `distances`,
    the function can provide 3D points (instead of rays).
    """
    # Sanity checks.
    if type(intrinsics_matrix) is not np.ndarray:
        raise TypeError(
            f"Camera intrinsics matrix should be an ND array. Got type {type(intrinsics_matrix)}."
        )

    if intrinsics_matrix.shape != (3, 3):
        raise ValueError(
            f"Expected an intrinsics matrix with shape (3, 3), but got {intrinsics_matrix.shape}."
        )

    if type(camera_pose) is not np.ndarray:
        raise TypeError(
            f"Camera pose should be an ND array. Got type {type(camera_pose)}."
        )

    # Prepare points for projection.
    points = prepare_points_for_projection(points=points, D=2)

    # Project the points into the world frame.
    camera_R, camera_t = camera_pose[:3, :3].reshape(3, 3), camera_pose[:3, 3].reshape(
        3, 1
    )
    projected_points = camera_R @ np.linalg.inv(intrinsics_matrix) @ points.transpose()
    projected_points = (distances * projected_points) + camera_t

    return np.transpose(projected_points)


def to_homogeneous_coords(points: np.ndarray) -> np.ndarray:
    """
    Converts a set of points to homogeneous coordinates. Assume points have shape (N, D), where N is the number of points and D is the
    dimension of each point.
    """
    return np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))


def to_nonhomogeneous_coords(points: np.ndarray) -> np.ndarray:
    """
    Converts a set of points to non-homogeneous coordinates. Assumes points have shape (N, D + 1), where N is the number of points and D is the dimension of each point.
    """
    return points[:, :-1] / points[:, -1].reshape(-1, 1)


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
