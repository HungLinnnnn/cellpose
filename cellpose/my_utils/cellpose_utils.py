import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from typing import Tuple, Optional
import os
from copy import deepcopy
from tqdm.auto import tqdm
from tqdm import trange
import warnings
warnings.simplefilter('default', UserWarning)  # 確保 UserWarning 顯示

from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.color import label2rgb, hsv2rgb
from skimage.segmentation import relabel_sequential
from skimage import graph

from shapely.geometry import Polygon

import torch

from ultralytics import SAM
from ultralytics.models.sam.amg import remove_small_regions

from cellpose import models


def reinhard_color_transfer(src_rgb, ref_rgb, verbose=False):
    """
    將 src_rgb 影像的色彩轉換成與 ref_rgb 相似的風格（Reinhard 方法）。
    
    參數:
      src_bgr: np.ndarray, uint8, RGB 格式的來源影像
      ref_bgr: np.ndarray, uint8, RGB 格式的參考影像
    
    回傳:
      transfer_bgr: np.ndarray, uint8, RGB 格式的輸出影像
    """
    # 1. 轉到 Lab 色彩空間
    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 2. 計算每個通道的均值與標準差
    src_means, src_stds = cv2.meanStdDev(src_lab)
    ref_means, ref_stds = cv2.meanStdDev(ref_lab)

    # 3. 進行通道級的線性變換
    #    out = (std_ref / std_src) * (src - mean_src) + mean_ref
    transfer_lab = src_lab.copy()
    for i in range(3):
        transfer_lab[..., i] = ((transfer_lab[..., i] - src_means[i]) *
                                 (ref_stds[i] / (src_stds[i] + 1e-8)) +
                                 ref_means[i])

    # 4. 裁剪到合法範圍並轉回 uint8
    
    if verbose:
        print(f"小於0的數目  : {(transfer_lab<0).astype(int).sum()}")
        print(f"大於255的數目: {(transfer_lab>255).astype(int).sum()}")

    transfer_lab = np.clip(transfer_lab, 0, 255).astype(np.uint8)
    transfer_rgb = cv2.cvtColor(transfer_lab, cv2.COLOR_LAB2RGB)

    return transfer_rgb


def plt_show_opencv(images:Tuple[np.ndarray], grid_size: Optional[Tuple[int, int]]=None, fsize: Optional[Tuple[int, int]]=None, title: Optional[Tuple[str]]=None, save_name: Optional[str]=None, full_image_title: str = None):
    
    if(len(images)) == 1:
        if len(images[0].shape) == 2:
            plt.imshow(images[0], cmap='gray')
        else:
            plt.imshow(images[0])
        
    else:
        if grid_size is None:
            num_image = len(images)
            num_row, num_col = math.ceil(num_image / 2), 2
            grid_size = (num_row, num_col)
            fsize = (num_col*8, num_row*8)
            
        num_images = len(images)
        num_grids = grid_size[0] * grid_size[1]
        if num_grids < num_images:
            raise ValueError(f"Number of grids({num_grids}) should be greater than number of images({num_grids}).")
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=fsize)
        axes = axes.ravel()
        for i in range(num_grids):
            axes[i].axis("off")
            
            if i < num_images and images[i] is not None:
                if len(images[i].shape) == 2:
                    axes[i].imshow(images[i], cmap='gray')
                else:
                    axes[i].imshow(images[i])

                if title is not None and title[i] is not None:
                    axes[i].set_title(title[i])

            
        
        if full_image_title is not None:
            fig.suptitle(full_image_title)
        
        plt.tight_layout()
        
    
    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


def plt_show_cmap(images:Tuple[np.ndarray], grid_size: Optional[Tuple[int, int]]=None, fsize: Optional[Tuple[int, int]]=None, title: Optional[Tuple[str]]=None, save_name: Optional[str]=None):
    
    if(len(images)) == 1:
        if len(images[0].shape) == 2:
            plt.imshow(images[0], cmap='viridis')
        else:
            plt.imshow(images[0])
        
    else:
        if grid_size is None:
            num_image = len(images)
            num_row, num_col = math.ceil(num_image / 2), 2
            grid_size = (num_row, num_col)
            fsize = (num_col*4, num_row*4)
            
        num_images = len(images)
        num_grids = grid_size[0] * grid_size[1]
        if num_grids < num_images:
            raise ValueError(f"Number of grids({num_grids}) should be greater than number of images({num_grids}).")
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=fsize)
        axes = axes.ravel()
        for i in range(num_grids):
            axes[i].axis("off")
            
            if i < num_images and images[i] is not None:
                if len(images[i].shape) == 2:
                    axes[i].imshow(images[i], cmap='viridis')
                else:
                    raise ValueError("要是單通道")

                if title is not None and title[i] is not None:
                    axes[i].set_title(title[i])

            
        
        plt.tight_layout()
        
    
    if save_name is not None:
        plt.savefig(save_name)

    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mask2rgb(masks_pred, image=None):
    """
    Return: rgb_mask, result
    """
    
    num_labels = masks_pred.max()
    rng = np.random.RandomState(42)
    hues = rng.rand(num_labels)            # 色相 0–1
    sats = rng.uniform(0.4, 0.9, num_labels)  # 飽和度 0.6–0.9
    vals = rng.uniform(0.5, 0.9, num_labels)  # 亮度    0.7–0.9

    # HSV -> RGB
    colors_hsv = np.stack([hues, sats, vals], axis=1)
    # skimage.color.hsv2rgb expects shape (..., 3)
    colors_rgb = hsv2rgb(colors_hsv[np.newaxis, :, :])[0]

    tmp_rgb = label2rgb(masks_pred, bg_label=0, colors=colors_rgb, kind="overlay")
    
    result = None
    if image is not None:
        image_for_plot = image.copy()
        image_for_plot[masks_pred>0] = (tmp_rgb[masks_pred>0] * 255).astype(np.uint8)
        image_weight = 0.4
        result = cv2.addWeighted(image, image_weight, image_for_plot, (1-image_weight), 0)

    
    return tmp_rgb, result


def merge_instances(
    maskA: np.ndarray,
    maskB: np.ndarray,
    probA: np.ndarray,
    probB: np.ndarray,
    contain_thresh: float = 0.9,
    contain_thresh_rev: float = 0.9,
    overlap_min: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    合併 A、B 兩個實例分割結果並重編號，同步回傳合併後的像素機率圖。

    參數：
    - maskA, maskB: (H, W) 整數 ndarray，0 為背景，正整數為不同實例
    - probA, probB: (H, W) float ndarray，模型對每個像素為「細胞」的機率
    - contain_thresh: B_j 被 A_i 包含的比例閾值 (|B_j∩A_i|/|B_j| ≥ contain_thresh)
    - contain_thresh_rev: A_j 被 B_i 包含的比例閾值 (|A_j∩B_i|/|A_j| ≥ contain_thresh_rev)
    - overlap_min: IoU ≥ overlap_min 時進行機率比較 (只針對重疊區域)

    回傳：
    - maskOut: (H, W) 整數 ndarray，重編號後的合併實例標籤
    - probOut: (H, W) float ndarray，合併後每個像素的機率值
    """

    H, W = maskA.shape

    # 1. 用 regionprops 取得所有實例的屬性
    propsA = regionprops(maskA)
    propsB = regionprops(maskB)
    dictA = {p.label: p for p in propsA}
    dictB = {p.label: p for p in propsB}

    # 2. 建立要刪除的標籤集合
    removeA = set()
    removeB = set()

    # 3. A 包含 B
    for a_label, a_reg in dictA.items():
        min_rA, min_cA, max_rA, max_cA = a_reg.bbox
        for b_label, b_reg in dictB.items():
            if b_label in removeB:
                continue
            min_rB, min_cB, max_rB, max_cB = b_reg.bbox
            # bbox 無交集 -> 跳過
            if max_rA <= min_rB or max_rB <= min_rA or max_cA <= min_cB or max_cB <= min_cA:
                continue
            # 計算 B_j 在 A_i 內的像素數
            rowsB, colsB = b_reg.coords[:,0], b_reg.coords[:,1]
            inter_count = np.sum(maskA[rowsB, colsB] == a_label)
            if inter_count / b_reg.area >= contain_thresh:
                removeB.add(b_label)

    # 4. B 包含 A
    for b_label, b_reg in dictB.items():
        min_rB, min_cB, max_rB, max_cB = b_reg.bbox
        for a_label, a_reg in dictA.items():
            if a_label in removeA or b_label in removeB:
                continue
            min_rA, min_cA, max_rA, max_cA = a_reg.bbox
            if max_rA <= min_rB or max_rB <= min_rA or max_cA <= min_cB or max_cB <= min_cA:
                continue
            rowsA, colsA = a_reg.coords[:,0], a_reg.coords[:,1]
            inter_count = np.sum(maskB[rowsA, colsA] == b_label)
            if inter_count / a_reg.area >= contain_thresh_rev:
                removeA.add(a_label)

    # 5. 部分重疊但不包含，IoU + 機率比較
    for a_label, a_reg in dictA.items():
        if a_label in removeA:
            continue
        min_rA, min_cA, max_rA, max_cA = a_reg.bbox
        for b_label, b_reg in dictB.items():
            if b_label in removeB:
                continue
            min_rB, min_cB, max_rB, max_cB = b_reg.bbox
            if max_rA <= min_rB or max_rB <= min_rA or max_cA <= min_cB or max_cB <= min_cA:
                continue
            # 計算重疊 bbox
            r0, r1 = max(min_rA, min_rB), min(max_rA, max_rB)
            c0, c1 = max(min_cA, min_cB), min(max_cA, max_cB)
            subA = maskA[r0:r1, c0:c1] == a_label
            subB = maskB[r0:r1, c0:c1] == b_label
            inter = np.logical_and(subA, subB).sum()
            if inter == 0:
                continue
            union = a_reg.area + b_reg.area - inter
            iou = inter / union
            if iou >= overlap_min:
                # 只在重疊區域計算平均機率
                coords = np.argwhere(np.logical_and(maskA == a_label, maskB == b_label))
                rows, cols = coords[:,0], coords[:,1]
                meanA = probA[rows, cols].mean()
                meanB = probB[rows, cols].mean()
                if meanA >= meanB:
                    removeB.add(b_label)
                else:
                    removeA.add(a_label)
                break

    # 6. 計算保留標籤集
    keepA = set(dictA.keys()) - removeA
    keepB = set(dictB.keys()) - removeB

    # 7. 重新編號並生成 maskOut、probOut
    maskOut = np.zeros((H, W), dtype=np.int32)
    probOut = np.zeros((H, W), dtype=probA.dtype)
    new_label = 1

    # 填入 A 的實例
    for a_label in sorted(keepA):
        region = (maskA == a_label)
        maskOut[region] = new_label
        probOut[region] = probA[region]  # 將 A 模型的機率貼過來
        new_label += 1

    # 填入 B 的實例（僅在未被 A 佔用的像素）
    for b_label in sorted(keepB):
        region = (maskB == b_label) & (maskOut == 0)
        maskOut[region] = new_label
        probOut[region] = probB[region]  # 將 B 模型的機率貼過來
        new_label += 1

    return maskOut, probOut

def merge_instances_area(
    maskA: np.ndarray,
    maskB: np.ndarray,
    probA: np.ndarray,
    probB: np.ndarray,
    contain_thresh: float = 0.9,
    contain_thresh_rev: float = 0.9,
    overlap_min: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    合併 A、B 兩個實例分割結果並重編號，同步回傳合併後的像素機率圖。

    參數：
    - maskA, maskB: (H, W) 整數 ndarray，0 為背景，正整數為不同實例
    - probA, probB: (H, W) float ndarray，模型對每個像素為「細胞」的機率
    - contain_thresh: B_j 被 A_i 包含的比例閾值 (|B_j∩A_i|/|B_j| ≥ contain_thresh)
    - contain_thresh_rev: A_j 被 B_i 包含的比例閾值 (|A_j∩B_i|/|A_j| ≥ contain_thresh_rev)
    - overlap_min: IoU ≥ overlap_min 時進行「取面積較大者」的最低重疊門檻

    回傳：
    - maskOut: (H, W) 整數 ndarray，重編號後的合併實例標籤
    - probOut: (H, W) float ndarray，合併後每個像素的機率值
    """

    H, W = maskA.shape

    # 1. 用 regionprops 取得所有實例的屬性
    propsA = regionprops(maskA)
    propsB = regionprops(maskB)
    dictA = {p.label: p for p in propsA}
    dictB = {p.label: p for p in propsB}

    # 2. 建立要刪除的標籤集合
    removeA = set()
    removeB = set()

    # 3. A 包含 B
    for a_label, a_reg in dictA.items():
        min_rA, min_cA, max_rA, max_cA = a_reg.bbox
        for b_label, b_reg in dictB.items():
            if b_label in removeB:
                continue
            min_rB, min_cB, max_rB, max_cB = b_reg.bbox
            # bbox 無交集 -> 跳過
            if max_rA <= min_rB or max_rB <= min_rA or max_cA <= min_cB or max_cB <= min_cA:
                continue
            # 計算 B_j 在 A_i 內的像素數
            rowsB, colsB = b_reg.coords[:,0], b_reg.coords[:,1]
            inter_count = np.sum(maskA[rowsB, colsB] == a_label)
            if inter_count / b_reg.area >= contain_thresh:
                removeB.add(b_label)

    # 4. B 包含 A
    for b_label, b_reg in dictB.items():
        min_rB, min_cB, max_rB, max_cB = b_reg.bbox
        for a_label, a_reg in dictA.items():
            if a_label in removeA or b_label in removeB:
                continue
            min_rA, min_cA, max_rA, max_cA = a_reg.bbox
            if max_rA <= min_rB or max_rB <= min_rA or max_cA <= min_cB or max_cB <= min_cA:
                continue
            rowsA, colsA = a_reg.coords[:,0], a_reg.coords[:,1]
            inter_count = np.sum(maskB[rowsA, colsA] == b_label)
            if inter_count / a_reg.area >= contain_thresh_rev:
                removeA.add(a_label)

    # 5. 部分重疊但不包含，依面積大小保留較大者
    for a_label, a_reg in dictA.items():
        if a_label in removeA:
            continue
        min_rA, min_cA, max_rA, max_cA = a_reg.bbox
        for b_label, b_reg in dictB.items():
            if b_label in removeB:
                continue
            min_rB, min_cB, max_rB, max_cB = b_reg.bbox
            # bbox 無交集 -> 跳過
            if max_rA <= min_rB or max_rB <= min_rA or max_cA <= min_cB or max_cB <= min_cA:
                continue
            # 計算重疊 bbox
            r0, r1 = max(min_rA, min_rB), min(max_rA, max_rB)
            c0, c1 = max(min_cA, min_cB), min(max_cA, max_cB)
            subA = maskA[r0:r1, c0:c1] == a_label
            subB = maskB[r0:r1, c0:c1] == b_label
            inter = np.logical_and(subA, subB).sum()
            if inter == 0:
                continue
            union = a_reg.area + b_reg.area - inter
            iou = inter / union
            # 若 IoU 達到門檻，保留面積較大者
            if iou >= overlap_min:
                if a_reg.area >= b_reg.area:
                    removeB.add(b_label)
                else:
                    removeA.add(a_label)
                break

    # 6. 計算保留標籤集
    keepA = set(dictA.keys()) - removeA
    keepB = set(dictB.keys()) - removeB

    # 7. 重新編號並生成 maskOut、probOut
    maskOut = np.zeros((H, W), dtype=np.int32)
    probOut = np.zeros((H, W), dtype=probA.dtype)
    new_label = 1

    # 填入 A 的實例
    for a_label in sorted(keepA):
        region = (maskA == a_label)
        maskOut[region] = new_label
        probOut[region] = probA[region]  # 將 A 模型的機率貼過來
        new_label += 1

    # 填入 B 的實例（僅在未被 A 佔用的像素）
    for b_label in sorted(keepB):
        region = (maskB == b_label) & (maskOut == 0)
        maskOut[region] = new_label
        probOut[region] = probB[region]  # 將 B 模型的機率貼過來
        new_label += 1

    return maskOut, probOut



def get_centroids_and_nearest_pixels(mask: np.ndarray):
    """
    計算 instance segmentation mask 中，每個 instance 的質心 (x, y) 與距質心最近的像素點 (x, y)。
    
    參數
    ----
    mask : np.ndarray
        一張 2D 整數陣列，0 為背景，其餘正整數為不同 instance id。
    
    回傳
    ----
    centroids : list
        長度為 (max_id + 1) 的 list，centroids[i] 對應 instance id = i：
        - centroids[0] = None
        - centroids[i] = (x, y)，x, y 為浮點數質心座標
    nearest_pixels : list
        長度同上，nearest_pixels[i] 對應 instance id = i：
        - nearest_pixels[0] = None
        - nearest_pixels[i] = (x, y)，x, y 為最接近質心的整數像素座標
    """
    props = regionprops(mask)
    max_id = int(mask.max())
    centroids = [None] * (max_id + 1)
    nearest_pixels = [None] * (max_id + 1)
    
    for prop in props:
        inst_id = prop.label
        # skimage 回傳 (row, col) 的浮點質心
        cy, cx = prop.centroid
        
        # 所有屬於該 instance 的像素座標 (row, col)
        coords = prop.coords  # shape=(N,2)
        
        # 計算每個像素到質心的歐氏距離
        dists = np.linalg.norm(coords - np.array([cy, cx]), axis=1)
        idx = np.argmin(dists)
        nearest_row, nearest_col = coords[idx]
        
        # 轉成 (x, y) 格式
        centroids[inst_id] = (cx, cy)
        nearest_pixels[inst_id] = (nearest_col, nearest_row)
    
    return centroids, nearest_pixels


def point_coordinate_transform(ori_shape, final_shape, input_point):
    input_point = deepcopy(input_point)
    labels = []
    for object_id in range(len(input_point)):
        label_point = []
        for point_id in range(len(input_point[object_id])):
            ori_x, ori_y = input_point[object_id][point_id]
            x = int(ori_x * (final_shape[0]/ori_shape[0]))
            y = int(ori_y * (final_shape[1]/ori_shape[1]))
            input_point[object_id][point_id][0] = x
            input_point[object_id][point_id][1] = y
            label_point.append(1)
        labels.append(label_point)

    return input_point, labels

def box_coordinate_transform(ori_shape, final_shape, input_box):
    """
    [
        [x1, y1, x2, y2],
        [x3, y3, x4, y4]
    ]
    """
    input_box = deepcopy(input_box)
    for box in input_box:
        for i in range(4):
            if i % 2 == 0:
                # x 
                box[i] = int(box[i] * (final_shape[0]/ori_shape[0]))
            else:
                # y
                box[i] = int(box[i] * (final_shape[1]/ori_shape[1]))
    
    return input_box


def polygon_iou(poly1_pts, poly2_pts):
    p1 = Polygon(poly1_pts).buffer(0)  # .buffer(0) => 會自動修復不合法的instance
    p2 = Polygon(poly2_pts).buffer(0)
    if not p1.is_valid or not p2.is_valid:
        # raise ValueError("输入的点序列必须构成有效多边形")
        return None
    inter_area = p1.intersection(p2).area
    # union_area = p1.union(p2).area
    
    # 計算重疊佔各自面積的比例
    ratio1 = inter_area / p1.area  # A 中有多少比例被 B 覆蓋
    ratio2 = inter_area / p2.area  # B 中有多少比例被 A 覆蓋
    
    # ratio1 與 ratio2 分別對應到「A 中被 B 覆蓋的比例」和「B 中被 A 覆蓋的比例」，最後取兩者的最大值作為函式結果。
    # 初衷：其中一方被覆蓋到一定比例即視為同一個細胞

    return max(ratio1, ratio2)

def merge_oversegmented_with_sam(
        image:np.ndarray,
        instance_mask: np.ndarray,
        sam_model: SAM,
        iou_threshold: float = 0.25,
        num_sample_points: int = 9,
        iterations: int = 1,
        return_detail_info=False
):
    """
    使用 SAM (Segment Anything Model) 來合併過度分割的實例。
    初衷：兩個instance mask的質心送入sam後，如果產生的mask會相交，代表他們很有可能本來就是同一個細胞

    參數:
        instance_mask (np.ndarray):
            原始的 instance segmentation mask，shape=(H, W)，每個像素值為實例 ID（0 為背景）。
        sam_model (SAM):
            已初始化且加載權重的 SAM 模型，用於生成 candidate mask 提示。
        iou_threshold (float, optional):
            當兩個 mask 的 IoU 高於此值且 Banana 分數低於 banana_threshold 時，才進行合併。預設為 0.5。
        return_detail_info (bool, optional):
            除了fianl_mask，是否回傳新增的mask, 新增數量, sam result

    返回:
        np.ndarray:
            經過處理後的 instance mask，同一過度分割片段已被合併，減少冗餘實例。
        np.ndarray:
            新增的instance mask
        int: 
            共新增幾個instance
        (註：新增指的是有幾個mask是從這個function產生的)
    """
    assert image.shape[0] == instance_mask.shape[0], "image, mask形狀要一樣"
    assert instance_mask.shape[0] == instance_mask.shape[1], "mask要是正方形"

    shape = instance_mask.shape

    # (舊作法)找質心，id i的質心在nearest_centroid[i]，因此nearest_centroid[0]是None (背景不定義質心)
    # centroid, nearest_centroid = get_centroids_and_nearest_pixels(instance_mask)
    # nearest_centroid = nearest_centroid[1:]
    # ori_points = [[list(p)] for p in nearest_centroid]

    for _ in range(iterations):
        # (新作法)改成一個instance sample幾個點送入sam產生mask
        sample_points_dict = sample_uniform_points_per_instance(instance_mask, num_sample_points)
        ori_points = [v for k, v in sample_points_dict.items()]

        # 產生送入sam的points, labels
        input_points, input_labels = point_coordinate_transform(
            shape,
            shape,
            ori_points
        )
        input_points, input_labels = np.array(input_points), np.array(input_labels)
        # 送入sam
        sam_results = sam_model(image, points=input_points, labels=input_labels, verbose=False)

        # 建立skimage graph rag
        edge_map = np.ones_like(instance_mask, dtype=float)  # 對我們的工作不重要，但還是要設定
        rag = graph.rag_boundary(instance_mask, edge_map=edge_map)  # 建立graph

        max_id = instance_mask.max()
        adj_list = []
        # 查看哪兩個id之間有相鄰
        for i in range(1, max_id+1):
            for j in range(1, i):
                if rag.has_edge(i, j):
                    adj_list.append((i, j))


        polygon_list = [None] + sam_results[0].masks.xy
        iou_list = []

        for adj in adj_list:
            
            iou = polygon_iou(polygon_list[adj[0]], polygon_list[adj[1]])
            iou_list.append(iou)

        num_merge = 0

        
        merge_mask = np.zeros_like(instance_mask, dtype=np.uint8)  # 只有新產生的mask
        new_mask = instance_mask.copy()  # 最終的instance mask (所有)

        for i in range(len(iou_list)):
            # 相交到一定程度，我就認為這兩個mask是屬於同一個細胞
            if iou_list[i] >= iou_threshold:
                id_x, id_y = adj_list[i]
                num_merge += 1
                new_mask[new_mask==id_y] = id_x
                merge_mask[instance_mask==id_x] = num_merge
                merge_mask[instance_mask==id_y] = num_merge
            

        new_mask, _, _ = relabel_sequential(new_mask)
        merge_mask, _, _ = relabel_sequential(merge_mask)

        instance_mask = new_mask.copy()  # for new iteration

    if return_detail_info:
        return new_mask, merge_mask, num_merge, sam_results
    else:
        return new_mask
    


def _merge_oversegmented(
        image:np.ndarray,
        instance_mask: np.ndarray,
        sam_results: list,
        iou_threshold: float = 0.1,
        return_detail_info=False
):
    """
    For測試的function，傳入sam_result，用以測試不同的sam結果
    使用 SAM (Segment Anything Model) 來合併過度分割的實例。
    初衷：兩個instance mask的質心送入sam後，如果產生的mask會相交，代表他們很有可能本來就是同一個細胞

    參數:
        instance_mask (np.ndarray):
            原始的 instance segmentation mask，shape=(H, W)，每個像素值為實例 ID（0 為背景）。
        sam_result (list):
            ultralytics sam_result
        iou_threshold (float, optional):
            當兩個 mask 的 IoU 高於此值且 Banana 分數低於 banana_threshold 時，才進行合併。預設為 0.5。
        return_detail_info (bool, optional):
            除了fianl_mask，是否回傳新增的mask, 新增數量, sam result

    返回:
        np.ndarray:
            經過處理後的 instance mask，同一過度分割片段已被合併，減少冗餘實例。
        np.ndarray:
            新增的instance mask
        int: 
            共新增幾個instance
        (註：新增指的是有幾個mask是從這個function產生的)
    """
    assert image.shape[0] == instance_mask.shape[0], "image, mask形狀要一樣"
    assert instance_mask.shape[0] == instance_mask.shape[1], "mask要是正方形"

    shape = instance_mask.shape

    # 建立skimage graph rag
    edge_map = np.ones_like(instance_mask, dtype=float)  # 對我們的工作不重要，但還是要設定
    rag = graph.rag_boundary(instance_mask, edge_map=edge_map)  # 建立graph

    max_id = instance_mask.max()
    adj_list = []
    # 查看哪兩個id之間有相鄰
    for i in range(1, max_id+1):
        for j in range(1, i):
            if rag.has_edge(i, j):
                adj_list.append((i, j))


    polygon_list = [None] + sam_results[0].masks.xy
    iou_list = []

    for adj in adj_list:
        
        iou = polygon_iou(polygon_list[adj[0]], polygon_list[adj[1]])
        iou_list.append(iou)

    num_merge = 0

    
    merge_mask = np.zeros_like(instance_mask, dtype=np.uint8)  # 只有新產生的mask
    new_mask = instance_mask.copy()  # 最終的instance mask (所有)

    for i in range(len(iou_list)):
        # 相交到一定程度，我就認為這兩個mask是屬於同一個細胞
        if iou_list[i] >= iou_threshold:
            id_x, id_y = adj_list[i]
            num_merge += 1
            new_mask[new_mask==id_y] = id_x
            merge_mask[instance_mask==id_x] = num_merge
            merge_mask[instance_mask==id_y] = num_merge
        

    new_mask, _, _ = relabel_sequential(new_mask)
    merge_mask, _, _ = relabel_sequential(merge_mask)

    if return_detail_info:
        return new_mask, merge_mask, num_merge, sam_results
    else:
        return new_mask





def sample_uniform_points_per_instance(mask: np.ndarray, n: int) -> dict:
    """
    在每個 instance 上近似平均分佈地取 n 個點（(x, y) 格式）。
    
    參數：
        mask (np.ndarray): 2D 整數陣列，shape = (H, W)，0 為背景，其他正整數為不同 instance id。
        n (int): 要為每個 instance 取的點數。必須為完全平方數，且每個 instance 的像素數 >= n。
    
    回傳：
        Dict[int, np.ndarray]: key = instance id，
            value = shape=(n, 2) 的整數陣列，每列為一個 (x, y) 座標，
            其中 x 對應 col，y 對應 row。
    
    Raises：
        ValueError: 當 n 不是完全平方數，或某 instance 的總像素數 < n。
    """
    # 1. 檢查 n 是否為完全平方數
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"n ({n}) must be a perfect square")

    # 2. 找到所有 instance ids（排除 0）
    inst_ids = np.unique(mask)
    inst_ids = inst_ids[inst_ids != 0]

    result = {}
    for inst_id in inst_ids:
        # 3. 取出該 instance 的所有像素座標
        ys, xs = np.where(mask == inst_id)  # ys: row, xs: col
        num_pixels = xs.size
        

        coords = np.vstack([xs, ys]).T  # shape = (num_pixels, 2), each = [x, y]

        if num_pixels < n:
            selected = [tuple(coords[i]) for i in range(num_pixels)]
            if len(selected) < n:
                for _ in range(len(selected), n):
                    selected.append(tuple(coords[0]))
            result[int(inst_id)] = np.array(selected, dtype=int)
            warnings.warn(
                f"Instance {inst_id} has only {num_pixels} pixels, fewer than n={n}",  # 訊息
                UserWarning,                                      # 分類
                stacklevel=2                                             # 堆疊層級，用於顯示正確位置
            )
            continue
            #raise ValueError(f"Instance {inst_id} has only {num_pixels} pixels, fewer than n={n}")

        # 4. 計算 bounding box
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        # 5. 在 bbox 上切 side x side 網格
        x_bins = np.linspace(xmin, xmax + 1, side + 1, dtype=int)
        y_bins = np.linspace(ymin, ymax + 1, side + 1, dtype=int)

        selected = []
        # 6. 每個格子內隨機取一點
        for i in range(side):
            for j in range(side):
                x0, x1 = x_bins[i],   x_bins[i + 1]
                y0, y1 = y_bins[j],   y_bins[j + 1]
                # 找出落在此格且屬於該 instance 的索引
                in_cell = np.where(
                    (xs >= x0) & (xs < x1) &
                    (ys >= y0) & (ys < y1)
                )[0]
                if in_cell.size:
                    idx = np.random.choice(in_cell)
                    selected.append(tuple(coords[idx]))
                # 若已選足 n 個就跳出
                if len(selected) >= n:
                    break
            if len(selected) >= n:
                break

        # 7. 補足剩下的點（不重複）
        if len(selected) < n:
            rem = n - len(selected)
            # 建立可抽樣的候選集 (排除已選點)
            chosen_set = set(selected)
            # all coords 列表
            all_pts = [tuple(pt) for pt in coords]
            candidates = [pt for pt in all_pts if pt not in chosen_set]
            # 隨機抽 rem 個
            extra = np.random.choice(len(candidates), rem, replace=False)
            for idx in extra:
                selected.append(candidates[idx])

        # 8. 存進結果 dict
        result[int(inst_id)] = np.array(selected, dtype=int)

    return result


def list_folders_by_depth(root: str, depth: int):
    """
    給定 root 資料夾和層數 depth，列出所有在該層的資料夾絕對路徑。
    例如 depth=1 表示 root/*，depth=2 表示 root/*/*。
    
    Args:
        root (str): 起始的根目錄路徑
        depth (int): 要搜尋的資料夾層級
    
    Returns:
        List[str]: 符合指定層級的資料夾絕對路徑清單
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"指定的根目錄不存在: {root}")
    if not os.path.isdir(root):
        raise NotADirectoryError(f"指定路徑不是資料夾: {root}")
    if depth < 1:
        raise ValueError("depth 必須是大於等於 1 的整數")

    result = []

    def walk_level(current_path, current_depth):
        if current_depth == depth:
            if os.path.isdir(current_path):
                result.append(os.path.abspath(current_path))
            return
        try:
            for item in os.listdir(current_path):
                next_path = os.path.join(current_path, item)
                if os.path.isdir(next_path):
                    walk_level(next_path, current_depth + 1)
        except PermissionError:
            print(f"無法存取: {current_path}")
    
    walk_level(root, 0)
    return result

def create_tail_subfolder(src_path: str, dst_dir: str, i: int) -> str:
    """
    在 dst_dir 下建立從 src_path 尾端取 i 層的資料夾結構。
    例如 src_path = a/b/c/x/y/z 且 i=2，則建立 dst_dir/y/z。

    Args:
        src_path (str): 原始資料夾路徑
        dst_dir (str): 目標根資料夾
        i (int): 要保留的尾端資料夾層數

    Returns:
        str: 建立好的新資料夾絕對路徑
    """
    if not os.path.isdir(src_path):
        raise NotADirectoryError(f"來源不是有效資料夾: {src_path}")
    # if not os.path.isdir(dst_dir):
    #     raise NotADirectoryError(f"目標根目錄不是有效資料夾: {dst_dir}")
    if i < 1:
        raise ValueError("參數 i 必須為正整數")

    # 拆分路徑並取得尾端 i 層
    tail_parts = os.path.normpath(src_path).split(os.sep)[-i:]
    target_path = os.path.join(dst_dir, *tail_parts)

    os.makedirs(target_path, exist_ok=True)
    return os.path.abspath(target_path)













