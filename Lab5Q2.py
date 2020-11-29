from PIL import Image

import numpy as np

im = Image.open('subject01.centerlight.pgm')
pix_val = list(im.getdata())

image_size=np.array(pix_val)
