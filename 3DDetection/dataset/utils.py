import scipy.ndimage as si
import cv2
import numpy as np 
from matplotlib import pyplot as plt 

def gaussian3D(shape, sigma=1):
  m, n, o = [(ss - 1.) / 2. for ss in shape]
  z, y, x  = np.ogrid[-m:m + 1, -n:n + 1, -o:o +1]

  h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h

def draw_umich_gaussian(heatmap, center, whd, radius, k=1):

    radius = gaussian_radius(whd)
    diameter = 2 * radius + 1
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter/6)

    x, y, z = int(center[0]), int(center[1]), int(center[2])

    height, width, depth = heatmap.shape[0:3]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    up, down = min(z, radius), min(depth - z, radius + 1)

    masked_heatmap = heatmap[z - up:z + down, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - up:radius + down, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
      np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
  depth, height, width = det_size

  a1  = 1
  b1  = (height + width + depth)
  c1  = width * height * depth * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width + depth)
  c2  = (1 - min_overlap) * width * height * depth
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width + depth)
  c3  = (min_overlap - 1) * width * height * depth
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def gaussian_map(img, center, sigma):

    img[center[0], center[1], center[2]] = 1
    img = si.gaussian_filter(img, sigma=(sigma, sigma, sigma))
    return img