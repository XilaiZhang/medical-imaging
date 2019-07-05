from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
import os
import shutil

RESIZE = False
DISPLAY = False
OUTPUT_BOOL = True
# Configure Filter Thresholds
BLURNESS_THRESHOLD = 40    # not in use; Not accurate
blur_index = 50
X_THRESHOLD = 40    # Vertical Offset
Y_THRESHOLD = 35    # Horizontal Offset

ALIGN = True
CROP = True

# Configure Crop Positions
CROP_X1 = 150   # Left
CROP_X2 = 750   # Right
CROP_Y1 = 450   # Near Nose Tip
CROP_Y2 = 790   # Above Eyes

SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"


# INITIAL TEST
if DISPLAY and CROP and OUTPUT_BOOL:
    print("ERROR: Cannot set DISPLAY and CROP and OUTPUT_BOOL = True together!")
    print("Now setting DISPLAY to FALSE...")
    DISPLAY = False
    # print("Exiting...")
    # exit(1)


# Check blurness
def variance_of_laplacian(image, x, y, w, h):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    print(x,y,w,h, np.shape(image))
    gray = cv2.imread(image,0)[max(0,y):y+h, max(0,x):x+w]
    if DISPLAY:
        cv2.imshow("cropped", gray)
        cv2.waitKey(0)
    if len(gray) > 500:
        gray = imutils.resize(gray, width=500)
    # print(gray)
    # laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # exit(1)
    index = cv2.Laplacian(gray, cv2.CV_64F).var()
    return index

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
ap.add_argument("-i", "--input", required=True,
    help="path to input folder of images")
ap.add_argument("-o", "--output", required=True,
    help="path to output folder")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

INPUT_FOLDER = args["input"]
OUTPUT_FOLDER_BASIC = args["output"]
OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASIC,"Output")
DISCARD_FOLDER = os.path.join(OUTPUT_FOLDER_BASIC,"Discard")
if OUTPUT_BOOL:
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(DISCARD_FOLDER):
        os.makedirs(DISCARD_FOLDER)
FILE_LIST = os.listdir(INPUT_FOLDER)
temp_list1 = []
for dir in FILE_LIST:
    temp_list1.append(os.path.join(INPUT_FOLDER, dir))
FILE_LIST = temp_list1
temp_list = []
for dir in FILE_LIST:
    print(dir)
    if os.path.isdir(dir):
        print("DIR detected")
        for new_dir in os.listdir(dir):
            FILE_LIST.append(os.path.join(dir, new_dir))
    elif os.path.isfile(dir) and dir.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        print("FILE detected")
        temp_list.append(dir)
FILE_LIST = temp_list

print("Start filtering {} images.".format(len(FILE_LIST)))

for output_index,img_dir in enumerate(FILE_LIST):
    # load the input image
    print("Current Img:", img_dir)
    image = cv2.imread(img_dir)

    if RESIZE == True:
        image = imutils.resize(image, width=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # For now, we limit the numebr of faces in each image to 1
    if len(rects) != 1:
        print("ERROR: Invalid image: #rects:", len(rects))
        if DISPLAY:
            cv2.imshow("Output", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        continue

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
        if DISPLAY:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        size = 1
        green = 0
        red = 255

        shape1.append(tuple(shape[30]))        # Nose Tip
        shape1.append(tuple(shape[8]))         # Chin
        shape1.append(tuple(shape[36]))        # Left eye left corner
        shape1.append(tuple(shape[45]))        # Right eye right corne
        shape1.append(tuple(shape[48]))        # Left Mouth corner
        shape1.append(tuple(shape[54]))        # Right mouth corner

    # Check against standard face
    size = image.shape
    # 2D image points.
    image_points = np.array([
                                shape1[0],
                                shape1[1],
                                shape1[2],
                                shape1[3],
                                shape1[4],
                                shape1[5]
                            ], dtype="double")

    # 3D model standard points.
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
    center = shape1[0]  # Fix center to Nose Tip
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # print("Rotation Vector:\n {0}".format(rotation_vector))

    rotation_matrix, jacobian_matrix = cv2.Rodrigues(rotation_vector)

    # print("Rotation Matrix:\n {0}".format(rotation_matrix))

    rx, ry, rz = rotationMatrixToEulerAngles(rotation_matrix)
    dx = math.degrees(rx)
    dy = math.degrees(ry)
    dz = math.degrees(rz)
    print("Euler Angles:", dx, dy, dz)

    # print("Translation Vector:\n {0}".format(translation_vector))

    # blur_index = variance_of_laplacian(os.path.join(INPUT_FOLDER, img_dir), x,y,w,h)
    # print("Blurness:", blur_index)

    print("TEST RESULT:")
    print("abs(dy)={}; 180-abs(dx)={}; blur={}".format(abs(dy), 180-abs(dx), blur_index))

    if DISPLAY:
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        # Display image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # Copy the eligible photos into the OUTPUT folder
    if abs(dy) < Y_THRESHOLD and 180 - abs(dx) < X_THRESHOLD and blur_index > BLURNESS_THRESHOLD:

        print("IMAGE PASSED TEST")
        if not CROP and OUTPUT_BOOL:
            # shutil.copyfile(os.path.join(INPUT_FOLDER, img_dir), os.path.join(OUTPUT_FOLDER, img_dir))
            shutil.copyfile(img_dir, os.path.join(OUTPUT_FOLDER, str(output_index) + '_' + os.path.basename(img_dir)))
            continue
    elif OUTPUT_BOOL:
        # shutil.copyfile(os.path.join(INPUT_FOLDER, img_dir), os.path.join(DISCARD_FOLDER, img_dir))
        shutil.copyfile(img_dir, os.path.join(DISCARD_FOLDER, str(output_index) + '_' + os.path.basename(img_dir)))
        continue
    else:
        continue

    if ALIGN:
        # Calculate Rotation Angle
        pts1 = np.array([
                            image_points[0],
                            image_points[2],
                            image_points[3]
                        ], dtype="float32")
        pts2 = np.array([
                            [450.0,450.0],
                            [225.0,620.0],
                            [675.0,620.0]
                        ], dtype="float32")

        scale = (pts1[2][0]-pts1[1][0])/450
        pts3 = pts2*scale

        transform_matrix = cv2.getAffineTransform(pts1,pts3)

        dst_img = np.flip(cv2.warpAffine(image, transform_matrix, (int((pts1[2][0]-pts1[1][0])*2),int((pts1[2][0]-pts1[1][0])*2))),0)

        if DISPLAY:
            # Display image
            cv2.imshow("Output", dst_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    if CROP:
        cropped_img = dst_img[(int((900-CROP_Y2)*scale)):(int((900-CROP_Y1)*scale)),int(CROP_X1*scale):(int(CROP_X2*scale))]

        # Draw a rectangle for cropping
        if DISPLAY:
            cv2.imshow("Output", cropped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        if OUTPUT_BOOL:
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, str(output_index) + '_' + os.path.basename(img_dir)), cropped_img)

print("\n\nDONE\n\n")
