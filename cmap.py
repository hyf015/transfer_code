import cv2, numpy as np
import sys
imname = sys.argv[1]
im = cv2.imread(imname,0)
r = np.ones_like(im) * 255
g = im.copy()
b = g.copy()
color = np.dstack((b,g,r))
cv2.imwrite('color_'+imname, color)
