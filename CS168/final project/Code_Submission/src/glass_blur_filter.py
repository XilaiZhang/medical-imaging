import torch
import os
import cv2
import argparse
import shutil
import numpy as np
import torchvision.models as models
from torch.autograd import Variable

SIZE = 224
BATCH_SIZE = 16

def img2matrix(src):
    img = cv2.imread(src)
    img = cv2.resize(img, (SIZE,SIZE))   # Standardize the size of the img (NOTE: Rescaling the img)
    swappedImg = np.zeros((3,SIZE,SIZE), dtype='float32')
    for i in range(3):
        swappedImg[i] = img[:,:,i]

    return swappedImg

def applyModel(model, data):
    model.eval()
    max_size = len(data)
    result_list = []
    for i in range(0, max_size, BATCH_SIZE):
        inputs = data[i:min(i+BATCH_SIZE, max_size)]
        inputs = Variable(inputs).float().cuda()
        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()

        for item in outputs:
            if item[0] > 0:
                result_list.append(1)
            else:
                result_list.append(0)
    assert(len(data) == len(result_list))

    return result_list


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input folder of images")
ap.add_argument("-o", "--output", required=True,
    help="path to output folder")
args = vars(ap.parse_args())

print("Start checking output directories")

INPUT_FOLDER = args["input"]
OUTPUT_FOLDER_BASIC = args["output"]
OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_BASIC,"Output")
DISCARD_FOLDER = os.path.join(OUTPUT_FOLDER_BASIC,"Discard")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(DISCARD_FOLDER):
    os.makedirs(DISCARD_FOLDER)
temp_list = os.listdir(INPUT_FOLDER)
FILE_LIST = []

# Remove extremely small images
#DEBUG: Potential Improvements here: Read images batch by batch to reduce memory usage
for img in temp_list:
    if os.stat(os.path.join(INPUT_FOLDER,img)).st_size >= 8000:
        FILE_LIST.append(img)

print("Start loading images")

# Load images
image_list = []
for img in FILE_LIST:
    image_list.append(img2matrix(os.path.join(INPUT_FOLDER,img)))

image_list = torch.Tensor(image_list)

# Apply Filters
print("Start applying filters")
model = torch.load("glass_filter_backup/full_model")
model = model.cuda()
print("Processing glasses filter")
result_list_1 = applyModel(model, image_list)

model = torch.load("blur_filter_backup/full_model")
model = model.cuda()
print("Processing blur filter")
result_list_2 = applyModel(model, image_list)

print("Filtration complete. Writing outputs.")

for i in range(len(FILE_LIST)):
    if result_list_1[i] or result_list_2[i]:
        shutil.copyfile(os.path.join(INPUT_FOLDER, FILE_LIST[i]), os.path.join(DISCARD_FOLDER, FILE_LIST[i]))
    else:
        shutil.copyfile(os.path.join(INPUT_FOLDER, FILE_LIST[i]), os.path.join(OUTPUT_FOLDER, FILE_LIST[i]))

print("\n\nDONE\n\n")
