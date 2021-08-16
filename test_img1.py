# импорт библиотек
from PIL import Image
import pytesseract
import sys
from pdf2image import convert_from_path
import os
# для поиска таблиц
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

MORPH = 7
CANNY = 250

_width  = 600.0
_height = 420.0
_margin = 0.0

corners = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)

pts_dst = np.array( corners, np.float32 )

img = 'test\\test.png'
im1 = cv2.imread(img)
im = cv2.imread(img)

gray = cv2.cvtColor( im1, cv2.COLOR_BGR2GRAY )
gray = cv2.bilateralFilter( gray, 1, 10, 120 )
edges  = cv2.Canny( gray, 10, CANNY )
kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( MORPH, MORPH ) )
closed = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )
contours, h = cv2.findContours( closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
for cont in contours:
    if cv2.contourArea( cont ) > 5000 :
        arc_len = cv2.arcLength( cont, True )
        approx = cv2.approxPolyDP( cont, 0.1 * arc_len, True )
        if ( len( approx ) == 2 ):
            IS_FOUND = 1
            #M = cv2.moments( cont )
            #cX = int(M["m10"] / M["m00"])
            #cY = int(M["m01"] / M["m00"])
            #cv2.putText(rgb, "Center", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            pts_src = np.array( approx, np.float32 )

            h, status = cv2.findHomography( pts_src, pts_dst )
            out = cv2.warpPerspective( im1, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )

            cv2.drawContours( im1, [approx], -1, ( 255, 0, 0 ), 2 )

cv2.imwrite('im1.jpg',im1)
cv2.imwrite('out.jpg',out)