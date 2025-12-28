import numpy as np
from cellpose import models, core, io, plot, utils
import matplotlib.pyplot as plt
import cv2
import math
from typing import Tuple, Optional
import os
from copy import deepcopy
from tqdm.auto import tqdm
from tqdm import trange

import torch

from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops_table
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.color import hsv2rgb

from ultralytics import SAM
from ultralytics.models.sam.amg import remove_small_regions

from .cellpose_utils import (
    sigmoid, 
    mask2rgb, 
    merge_instances_area,
    merge_oversegmented_with_sam
)


def cellpose_multi_scale_predict(image_list, size_list, model=None, **kwargs):
    """
    Input: 
    - image_list: ndarray形式image list
    - size_list: 想要inference的size
    - model: cellpose model
    - kwargs: 限定要傳入cellpose_model.eval()的其他參數

    Return:
    - result_list
    - masks_list
    """

    assert model is not None, "請傳入指定cellpose model"

    result_list = []
    masks_list = []
    
    for image in image_list:
        
        resize_image_list = [
            cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
            for size in size_list
        ]

        masks_pred_list = []
        flows_list = []

        for i, resize_image in enumerate(resize_image_list):

            masks_pred, flows, styles = model.eval(
                resize_image, 
                niter=1000,
                cellprob_threshold=-3,
                flow_threshold=3,
                min_size=int(300 * (size_list[i]/512)**2),
                **kwargs
            )
            masks_pred_list.append(masks_pred)
            flows_list.append(flows)
        # 
        mask_size = size_list[-1]
        mask_tmp = masks_pred_list[-1]
        prob_tmp = sigmoid(flows_list[-1][2])

        for i in range(len(size_list)-1):
            mask_a = cv2.resize(masks_pred_list[i], (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
            prob_a = cv2.resize(flows_list[i][2], (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)

            mask_tmp, prob_tmp = merge_instances_area(
                maskA=mask_a,
                maskB=mask_tmp,
                probA=sigmoid(prob_a),
                probB=prob_tmp,
                contain_thresh=0.6,
                contain_thresh_rev=0.6,
            )

        rgb, result = mask2rgb(mask_tmp, resize_image_list[-1])

        # cv2.imwrite("./experiment/tmp1.png", (rgb*255).astype(np.uint8))

        masks_list.append(mask_tmp)
        result_list.append(result)

    return result_list, masks_list


def cellpose_multi_scale_predict_sam_oversegment(
        image_list,
        size_list,
        cellpose_model=None,
        sam_model=None,
        iou_threshold=0.25,
        **kwargs
):
    """
    Input: 
    - image_list: ndarray形式image list
    - size_list: 想要inference的size
    - cellpose_model: cellpose model
    - sam_model: sam model
    - iou_threshold: 重疊多少視為同一個
    - kwargs: 限定要傳入cellpose_model.eval()的其他參數

    Return:
    - result_list
    - masks_list
    """

    sam_model = SAM("sam_l.pt") if sam_model is None else sam_model
    cellpose_model = models.CellposeModel(gpu=True) if cellpose_model is None else cellpose_model

    _, masks_list = cellpose_multi_scale_predict(
        image_list,
        size_list,
        cellpose_model,
        **kwargs
    )

    shape = masks_list[0].shape

    resize_image_list = [
        cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)
        for image in image_list
    ]

    result_list = []
    final_mask_list = []

    for resize_image, instance_mask in zip(resize_image_list, masks_list):
        final_mask = merge_oversegmented_with_sam(
            resize_image,
            instance_mask,
            sam_model,
            iou_threshold
        )

        result_list.append(mask2rgb(final_mask, resize_image)[1])
        final_mask_list.append(final_mask)
        #print(num_merge)

    return result_list, final_mask_list







if __name__ == '__main__':
    cellpose = models.CellposeModel(gpu=True)

    image_path = '/data1_19T/r12_jason/cell_seg/Dr_Tseng/20250525_Hs68/10X/Day 3/10ng_10X_Day 3_6.tif'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

    result_list, _ = cellpose_multi_scale_predict_sam_oversegment(
        [image],
        list(range(512, 1025, 128)),
        cellpose_model=cellpose
    )

    cv2.imwrite("./experiment/tmp.png", cv2.cvtColor(result_list[0], cv2.COLOR_RGB2BGR))