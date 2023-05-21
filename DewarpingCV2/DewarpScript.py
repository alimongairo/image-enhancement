import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped


def main(input_path, output_path):
	image = cv2.imread(input_path)
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height=500)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	break_flag = False
	for min_par in [i/100 for i in range(0, 80, 5)]:
		if break_flag:
			break
		for max_par in [i/10 for i in range(9, 20, 1)]:

			med_val = np.median(image)
			lower = int(min_par * med_val)
			upper = int(max_par * med_val)
			edged = cv2.Canny(blurred, lower, upper)

			cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

			screenCnt = []
			for c in cnts:
				peri = cv2.arcLength(c, True)
				approx = cv2.approxPolyDP(c, 0.02 * peri, True)
				if len(approx) == 4:
					screenCnt = approx
					break

			try:
				# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
				warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
				if warped.shape[0] < 4000 or warped.shape[1] < 4000:
					# image = orig.copy()
					continue
				else:
					cv2.imwrite(output_path, warped)
					break_flag = True
					break

			except:
				continue

	if len(screenCnt):
		warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
		cv2.imwrite(output_path, warped)
	else:
		print('Error: no bounds found')
		cv2.imwrite(output_path, orig)
