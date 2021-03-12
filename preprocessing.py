from skimage.transform import resize
from skimage.filters import gaussian
import numpy as np


img = "placeholder"
# Resize the image using scikit image
resize(image=img, output_shape=(512, 512))

# Create a random displacement map
raw_disp = np.random.uniform(low=-1000, high=1000, size=(512,512))
disp = gaussian(raw_disp,sigma=100)

def displace(im, disp_map):
    # Displaces an image using a displacement map. Code from https://stackoverflow.com/questions/57626940/is-there-a-function-which-apply-a-displacement-map-matrix-on-an-image
    im = np.asarray(im)
    disp_map = np.asarray(disp_map)
    grid = np.ogrid[list(map(slice, disp_map.shape[:-1]))]
    result = np.zeros_like(im)
    np.add.at(result, tuple((g + disp_map[..., i]) % im.shape[i]
                            for i, g in enumerate(grid)), im)
    return result

# Each pair of values represents the number of rows and columns
# that each element will be displaced
disp_map = np.array([[[ 0,  0], [ 0, -1], [ 0,  1], [ 0,  0]],
                     [[ 0,  0], [ 1,  0], [ 1,  1], [ 0,  0]],
                     [[ 0,  1], [ 0,  1], [ 0,  0], [ 0,  0]]])
im = np.array([[ 0,  1,  1,  0],
               [ 0,  1,  1,  0],
               [ 0,  0,  0,  0]])
output = displace(im, disp_map)
print(output)
# [[1 0 0 1]
#  [0 0 0 0]
#  [0 1 0 1]]