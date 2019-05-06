# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
RESIZE = True

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
# print(image)
# cv2.imshow("Output", image)
# cv2.waitKey(0)
# exit(1)
backup_img = image.copy()
if RESIZE == True:
	backup_img = imutils.resize(backup_img, width=500)
	image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
print("#rects:", len(rects))

shape1 = []

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.rectangle(backup_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	size = 1
	green = 0
	red = 255

	# for (x, y) in shape[54:]:
	# 	cv2.circle(image, (x, y), size, (0, green, red), -1)
	# 	size += 1
	# 	if size == 10:
	# 		size = 1
	# 		green += 10
	# 		red = 0

	# shape1.append(list(shape[30]))		# Nose Tip
	# shape1.append(list(shape[8]))		# Chin
	# shape1.append(list(shape[36]))		# Left eye left corner
	# shape1.append(list(shape[45]))		# Right eye right corne
	# shape1.append(list(shape[48]))		# Left Mouth corner
	# shape1.append(list(shape[54]))		# Right mouth corner

	shape1.append(tuple(shape[30]))		# Nose Tip
	shape1.append(tuple(shape[8]))		# Chin
	shape1.append(tuple(shape[36]))		# Left eye left corner
	shape1.append(tuple(shape[45]))		# Right eye right corne
	shape1.append(tuple(shape[48]))		# Left Mouth corner
	shape1.append(tuple(shape[54]))		# Right mouth corner


	# print(shape)
	# print(len(shape))
print(shape1)
# print(backup_img - image)
# exit(1)
# show the output image with the face detections + facial landmarks
# cv2.imshow("Output", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
for (x, y) in shape1:
	cv2.circle(backup_img, (x, y), 3, (0, 0, 225), -1)
cv2.imshow("Output", backup_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


###################################################################################
im = cv2.imread(args["image"])
if RESIZE == True:
	im = imutils.resize(im, width=500)
size = im.shape
#2D image points. If you change the image, you need to change vector
# image_points = np.array(shape1)
image_points = np.array([
							shape1[0],
							shape1[1],
							shape1[2],
							shape1[3],
							shape1[4],
							shape1[5]
						], dtype="double")
print(image_points)

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner

                        ])


# Camera internals

focal_length = size[1]
# focal_length = math.abs(math.sqrt(shape1[2][0] ** 2 + shape1[2][1] ** 2) - math.sqrt(shape1[3][0] ** 2 + shape1[3][1] ** 2)) * 5
# center = (size[1]/2, size[0]/2)
center = shape1[0]
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

print("Camera Matrix :\n {0}".format(camera_matrix))

dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

print("Rotation Vector:\n {0}".format(rotation_vector))
rotation_matrix, jacobian_matrix = cv2.Rodrigues(rotation_vector)
print("Rotation Matrix:\n {0}".format(rotation_matrix))

####################
rx, ry, rz = rotationMatrixToEulerAngles(rotation_matrix)
dx = math.degrees(rx)
dy = math.degrees(ry)
dz = math.degrees(rz)
print("Euler Angles:", dx, dy, dz)
####################

print("Translation Vector:\n {0}".format(translation_vector))


# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose


(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


p1 = ( int(image_points[0][0]), int(image_points[0][1]))
p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

cv2.line(im, p1, p2, (255,0,0), 2)

# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
