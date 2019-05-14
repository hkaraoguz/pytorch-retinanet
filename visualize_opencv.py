import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, \
    Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer

import skimage

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

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

    return new_image


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
    rncvimg = resize_image(ncvimg)
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

        dim = (rncvimg.shape[1]-32, rncvimg.shape[0]-32)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


        img[img < 0] = 0
        img[img > 255] = 255

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
            #print(label_name)

        cv2.imshow('img', resized)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()