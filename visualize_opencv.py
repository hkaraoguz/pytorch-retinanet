import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import json

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, \
    Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer

import skimage
import csv

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

home_path = os.getenv("HOME")
labels_path = os.path.join(home_path,"Dev","pytorch_retinanet","labels.csv")

labels = {}
with open(labels_path,"r") as f:
    line = f.readline()
    items = line.split(",")
    labels[int(items[1])] = items[0]
    while line :
        line = f.readline()
        items = line.split(",")
        if(len(items) !=2):
            break
        
        labels[int(items[1])] = items[0]


print(labels)

def normalize_image(img):
        
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
        
    img = (img.astype(np.float32)-mean)/std
    
    return img

def resize_image(image, min_side=608, max_side=1024):

    rows, cols, cns = image.shape

    print(rows," ",cols)

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows%32
    pad_h = 32 - cols%32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)

    return new_image, pad_w, pad_h


def inference(cvimg, conf_thresh=0.5):
    '''
    inference runs on an opencv image and returns the results in a json format
    '''
    results_dict = {}
    # cvimg = cv2.imread("data/val/shelves4.jpg",-1)
    #try:
    home_path = os.getenv("HOME")
    model_path = os.path.join(home_path,"Dev","pytorch_retinanet","model_final.pt")
    retinanet = torch.load(model_path)
    retinanet = retinanet.cuda()
    retinanet.eval()
    # Transform image into right format
    #print(cvimg)
    rgbcvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    #print("color")
    ncvimg = normalize_image(rgbcvimg.astype(np.float32)/255.0)
    rncvimg, pad_w, pad_h = resize_image(ncvimg)
    print("pad_w {} pad_h {}".format(pad_w, pad_h))
    #print(rncvimg.shape)
    img = torch.from_numpy(rncvimg).cuda().float()
    img = img[np.newaxis, :] 
    #print(img.shape)
    img = img.permute(0, 3, 1, 2)

    with torch.no_grad():
        st = time.time()
        scores, classification, transformed_anchors = retinanet(img)
        print('Elapsed time: {}'.format(time.time()-st))
        idxs = np.where(scores > conf_thresh)
        srcimg = cvimg.copy() 

        ratio_y = float(rncvimg.shape[0]-pad_w)/srcimg.shape[0]
        ratio_x = float(rncvimg.shape[1]-pad_h)/srcimg.shape[1]

        ratio_y = 1./ratio_y
        ratio_x = 1./ratio_x

        #dim = (rncvimg.shape[1]-pad_w, rncvimg.shape[0]-pad_h)
        # resize image
        #resized = cv2.resize(srcimg, dim, interpolation=cv2.INTER_AREA)
        
        bboxes = []
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0]*ratio_x)
            y1 = int(bbox[1]*ratio_y)
            x2 = int(bbox[2]*ratio_x)
            y2 = int(bbox[3]*ratio_y)
            label_name = labels[int(classification[idxs[0][j]])]
            #draw_caption(resized, (x1, y1, x2, y2), label_name)

            cv2.rectangle(srcimg, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            bboxes.append([x1, y1, x2, y2, label_name, scores[j].item()])
            #print(label_name)
        #results_dict["status"] = "Success"
        results_dict["result"] = bboxes
    #except Exception as ex:
    #    print(ex)
    #    results_dict["status"] = "Fail. {}".format(str(ex))
    #    results_dict["results"] = []
    
    torch.cuda.empty_cache()
    
    return bboxes


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1,
                                          drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1,
                                collate_fn=collater, batch_sampler=sampler_val)

    retinanet = torch.load(parser.model)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    
    cvimg = cv2.imread("data/val/shelves4.jpg",-1)
    rgbcvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    ncvimg = normalize_image(rgbcvimg.astype(np.float32)/255.0)
    rncvimg,pad_w,pad_h = resize_image(ncvimg)
    print("pad_w {} pad_h {}".format(pad_w,pad_h))
    print(rncvimg.shape)
    img = torch.from_numpy(rncvimg).cuda().float()
    img = img[np.newaxis, :] 
    #print(img.shape)
    img = img.permute(0, 3, 1, 2)

    with torch.no_grad():
        st = time.time()
        scores, classification, transformed_anchors = retinanet(img)
        print('Elapsed time: {}'.format(time.time()-st))
        idxs = np.where(scores > 0.5)
        img = cvimg.copy() #np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

        dim = (rncvimg.shape[1]-pad_w, rncvimg.shape[0]-pad_h)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


        #img[img < 0] = 0
        #img[img > 255] = 255

       # img = np.transpose(img, (1, 2, 0))

       # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset_val.labels[int(classification[idxs[0][j]])]
            draw_caption(resized, (x1, y1, x2, y2), label_name)

            cv2.rectangle(resized, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print(scores[j].item())
            #print(label_name)

        cv2.imshow('img', resized)
        cv2.waitKey(0)

#if __name__ == '__main__':
#    main()