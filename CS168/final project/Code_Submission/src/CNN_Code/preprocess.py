import json
import os
import numpy as np
import cv2
import pickle
import re

GLASS = False
BLUR = True

# Glasses Filter
Train_Annotation_Path = '/home/tony/Documents/glassdata/train_labels.txt'
Test_Annotation_Path = '/home/tony/Documents/glassdata/test_labels.txt'
Train_DST_Folder = '/home/tony/Documents/glassdata/train_dataset'
Train_GLS_Folder = '/home/tony/Documents/glassdata/train_dataset_g'
Train_NONGLS_Folder = '/home/tony/Documents/glassdata/train_dataset_ng'
Test_DST_Folder = '/home/tony/Documents/glassdata/test_dataset'
Test_GLS_Folder = '/home/tony/Documents/glassdata/test_dataset_g'
Test_NONGLS_Folder = '/home/tony/Documents/glassdata/test_dataset_ng'
TRAIN_SIZE = len(os.listdir(Train_GLS_Folder)) + len(os.listdir(Train_NONGLS_Folder))
TEST_SIZE = len(os.listdir(Test_GLS_Folder)) + len(os.listdir(Test_NONGLS_Folder))

def rename():
    count = 0
    for folder in [Train_NONGLS_Folder, Train_GLS_Folder, Test_NONGLS_Folder, Test_GLS_Folder]:
        filelist = os.listdir(folder)
        for file in filelist:
            if ' ' in file:
                count += 1
                src_filename = os.path.join(folder, file)
                new_file = file.replace(' ', '_')
                dst_filename = os.path.join(folder, new_file)
                os.rename(src_filename, dst_filename)
    print("Total Number of Changed Files: {}".format(count))

# Create a txt file consisting of all file info with labels
def processingLabel4Train():
    result = np.zeros((TRAIN_SIZE,2), dtype=object)
    iter = 0
    filelist1 = os.listdir(Train_GLS_Folder)
    for file in filelist1:
        # print(file)
        result[iter]=[file,1]
        iter = iter + 1
    filelist2 = os.listdir(Train_NONGLS_Folder)
    for file in filelist2:
        result[iter]=[file,-1]
        iter = iter + 1

    np.savetxt(Train_Annotation_Path, result, delimiter=' ', fmt="%s")

# Package the image with label info into pickle files
def processingData4Train():
    fid = open(Train_Annotation_Path, 'r')
    lines = fid.readlines()
    fid.close()
    print(fid)
    # annotations = list()
    for line in lines:
        line = line.replace('\n', '')
        oriTokens = line.split(' ')
        tokens = list()
        for token in oriTokens:
            if token != '':
                tokens.append(token)
        print(tokens)
        assert(len(tokens)==2)
        # annotations.append({'image_id': tokens[0], 'labels': tokens[1]})

        ori_folder = ""
        if tokens[1] == '1':
            ori_folder = Train_GLS_Folder
        elif tokens[1] == '-1':
            ori_folder = Train_NONGLS_Folder
        else:
            print(tokens)
            print("ERROR: Invalid label.")
            break

        img = cv2.imread(os.path.join(ori_folder, tokens[0]))
        # cv2.imshow("Output", img)
        # cv2.waitKey(0)
        img_id = ""
        if re.match(r".*\.png", tokens[0]) or tokens[0].endswith(".jpg"):
            img_id = tokens[0][0:-4]
        elif re.match(r".*\.jpeg", tokens[0]):
            img_id = tokens[0][0:-5]
        print(tokens, img_id)
        fid = open(os.path.join(Train_DST_Folder, img_id + '.pkl'), 'wb')
        pickle.dump([img, [tokens[1]]], fid)
        fid.close()


def processingLabel4Test():
        result = np.zeros((TEST_SIZE,2), dtype=object)
        iter = 0
        filelist1 = os.listdir(Test_GLS_Folder)
        for file in filelist1:
            # print(file)
            result[iter]=[file,1]
            iter = iter + 1
        filelist2 = os.listdir(Test_NONGLS_Folder)
        for file in filelist2:
            result[iter]=[file,-1]
            iter = iter + 1

        np.savetxt(Test_Annotation_Path, result, delimiter=' ', fmt="%s")

def processingData4Test():
    fid = open(Test_Annotation_Path, 'r')
    lines = fid.readlines()
    fid.close()
    print(fid)
    # annotations = list()
    for line in lines:
        line = line.replace('\n', '')
        oriTokens = line.split(' ')
        tokens = list()
        for token in oriTokens:
            if token != '':
                tokens.append(token)
        # print(tokens)
        assert(len(tokens)==2)
        # annotations.append({'image_id': tokens[0], 'labels': tokens[1]})

        ori_folder = ""
        if tokens[1] == '1':
            ori_folder = Test_GLS_Folder
        elif tokens[1] == '-1':
            ori_folder = Test_NONGLS_Folder
        else:
            print(tokens)
            print("ERROR: Invalid label.")
            break
        img = cv2.imread(os.path.join(ori_folder, tokens[0]))
        img_id = ""
        if re.match(r".*\.png", tokens[0]) or tokens[0].endswith(".jpg"):
            img_id = tokens[0][0:-4]
        elif re.match(r".*\.jpeg", tokens[0]):
            img_id = tokens[0][0:-5]
        print(tokens, img_id)
        fid = open(os.path.join(Test_DST_Folder, img_id + '.pkl'), 'wb')
        pickle.dump([img, [tokens[1]]], fid)
        fid.close()

if GLASS:
    rename()

    processingLabel4Train()
    processingData4Train()

    processingLabel4Test()
    processingData4Test()
    pass


# Recycle the functions and perform preprocessing for blurriness filter
# Blurriness Filter
Train_Annotation_Path = '/home/tony/Documents/blurdata/train_labels.txt'
Test_Annotation_Path = '/home/tony/Documents/blurdata/test_labels.txt'
Train_DST_Folder = '/home/tony/Documents/blurdata/train_dataset'
Train_GLS_Folder = '/home/tony/Documents/blurdata/train_dataset_b'
Train_NONGLS_Folder = '/home/tony/Documents/blurdata/train_dataset_nb'
Test_DST_Folder = '/home/tony/Documents/blurdata/test_dataset'
Test_GLS_Folder = '/home/tony/Documents/blurdata/test_dataset_b'
Test_NONGLS_Folder = '/home/tony/Documents/blurdata/test_dataset_nb'
TRAIN_SIZE = len(os.listdir(Train_GLS_Folder)) + len(os.listdir(Train_NONGLS_Folder))
TEST_SIZE = len(os.listdir(Test_GLS_Folder)) + len(os.listdir(Test_NONGLS_Folder))

if BLUR:
    rename()

    processingLabel4Train()
    processingData4Train()

    processingLabel4Test()
    processingData4Test()
    pass
