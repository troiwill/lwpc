from lwpc.model import PerspectiveCameraModel
import numpy as np


def create_world_points():
    return np.array(
        [
            [-0.1, -0.1, 1.0],
            [-0.1, 0.0, 1.0],
            [-0.1, 0.1, 1.0],
            [0.0, -0.1, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.1, -0.1, 1.0],
            [0.1, 0.0, 1.0],
            [0.1, 0.1, 1.0],
        ]
    )


def create_image_points():
    return np.array(
        [
            [490.0, 362.0],
            [490.0, 512.0],
            [490.0, 662.0],
            [640.0, 362.0],
            [640.0, 512.0],
            [640.0, 662.0],
            [790.0, 362.0],
            [790.0, 512.0],
            [790.0, 662.0],
        ]
    )


def create_intrinsics():
    focal_len = 0.015
    pixel_len = 10e-6
    fpr = focal_len / pixel_len
    return np.array([[fpr, 0.0, 640.0], [0.0, fpr, 512.0], [0.0, 0.0, 1.0]])


def test_project_world_to_image():
    intrinsics = create_intrinsics()
    model = PerspectiveCameraModel(intrinsics)

    points = create_world_points()
    camera_pose = np.eye(4)

    x = model.project_to_image(camera_pose=camera_pose, points=points)

    expected = create_image_points()
    assert np.allclose(x, expected)


def test_project_image_to_world():
    intrinsics = create_intrinsics()
    model = PerspectiveCameraModel(intrinsics)

    points = create_image_points()
    camera_pose = np.eye(4)

    distances = np.linalg.norm(create_world_points(), axis=1)
    x = model.project_to_world(
        camera_pose=camera_pose, points=points, distances=distances
    )

    expected = create_world_points()
    assert np.allclose(x, expected, atol=0.01)
