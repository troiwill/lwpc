# Light-Weight Perspective Camera Model

This module creates a simple, lightweight perspective camera model.

### Dependencies

If you will not install this package via the instructions below, you will need `NumPy >= 1.13.0`.

### Installation

Clone the repository and install this package via `build` and `pip`.
```
# Clone the repository.
cd <REPO DIR>
git clone https://github.com/troiwill/lwpc.git
cd lwpc

# Build and install package.
python3 -m build
python3 -m pip install dist/lwpc-*.whl
```

### Example

The code below shows how to use this package. The example was created using the "Robotics, Vision, and Control: Fundamental Algorithms in MATLAB (2nd Edition)" by Peter Corke using Sections 11.1.3 - 11.1.5.

**Note:** If you're using a real or simulated camera, you can get the intrinsics matrix using camera calibration.

```
import numpy as np
from lwpc.model import PerspectiveCameraModel

# Define the points in the world.
X = np.array([
    [-0.1, -0.1, 1. ],
    [-0.1,  0. , 1. ],
    [-0.1,  0.1, 1. ],
    [ 0. , -0.1, 1. ],
    [ 0. ,  0. , 1. ],
    [ 0. ,  0.1, 1. ],
    [ 0.1, -0.1, 1. ],
    [ 0.1,  0. , 1. ],
    [ 0.1,  0.1, 1. ]
])

# Define the intrinsics matrix.
focal_len = 0.015
pixel_len = 10e-6
fpr = focal_len / pixel_len
intrinsics_matrix = np.array([
    [ fpr, 0.0, 640.0 ],
    [ 0.0, fpr, 512.0 ],
    [ 0.0, 0.0,   1.0 ]
])

# Define the camera pose.
camera_pose = np.eye(4)

model = PerspectiveCameraModel(intrinsics_matrix)
x = model.project_to_image(camera_pose, X)

print(x)
```

Your results should be as follows:
```
[[490. 362.]
 [490. 512.]
 [490. 662.]
 [640. 362.]
 [640. 512.]
 [640. 662.]
 [790. 362.]
 [790. 512.]
 [790. 662.]]
 ```
