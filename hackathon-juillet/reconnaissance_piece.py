print("Hello")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import ndimage


img = image.imread("piece1.jpg")

print(img.shape)

img_gris = img.mean(axis=2)

seuil = 160  

img_seuil = np.where(img_gris > seuil, 0, 255)


masque = img_seuil > 0
masque_erode = ndimage.binary_erosion(masque)
bord = masque & ~masque_erode

plt.imshow(bord, cmap='gray')
plt.show()

print(bord)