# Copyright (c) V-DETR authors. All Rights Reserved.
import torch
import datetime
import logging
import math
import time
import sys
from tqdm import tqdm
import numpy as np
from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
    batch_dict_to_cuda,
)
from utils.box_util import (flip_axis_to_camera_tensor, get_3d_box_batch_tensor)
import json

def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
            curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
            and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
                (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        if args.lr_scheduler == 'cosine':
            # Cosine Learning Rate Schedule
            curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
                    1 + math.cos(math.pi * curr_epoch_normalized)
            )
        else:
            step_1, step_2 = args.step_epoch.split('_')
            step_1, step_2 = int(step_1), int(step_2)
            if curr_epoch_normalized < (step_1 / args.max_epoch):
                curr_lr = args.base_lr
            elif curr_epoch_normalized < (step_2 / args.max_epoch):
                curr_lr = args.base_lr / 10
            else:
                curr_lr = args.base_lr / 100
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
        args,
        curr_epoch,
        model,
        optimizer,
        criterion,
        dataset_config,
        dataset_loader,
):
    ap_calculator = None

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_data_label in tqdm(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        batch_data_label = batch_dict_to_cuda(batch_data_label, local_rank=net_device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        if args.use_superpoint:
            inputs["superpoint_per_point"] = batch_data_label["superpoint_labels"]
        outputs = model(inputs)
        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)

        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            eta_seconds = (max_iters - curr_iter) * (time.time() - curr_time)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            # print(
            #     f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; ETA {eta_str}"
            # )

        curr_iter += 1
        barrier()

    return ap_calculator, curr_iter, curr_lr, loss_avg.avg, loss_dict_reduced

def trans(bbox):
    bbox = bbox[:, :, [0, 2, 1]]
    bbox[:, :, 2] = -1 * bbox[:, :, 2]
    return bbox

def bbox_corners_to_xywlwh(bbox_corners):
    """
    将含有碰撞框八个顶点的numpy数组转换为【x,y,z,l,w,h】的列表

    参数:
    bbox_corners (np.ndarray): 形状为(N, 8, 3)的NumPy数组,包含了N个物体的8个顶点坐标(x, y, z)

    返回:
    list: 长度为N的列表,每个元素包含物体的中心坐标(x, y, z)以及长宽高(l, w, h)
    """

    corners = bbox_corners
    # 计算中心点坐标
    x = (corners[:, 0].min() + corners[:, 0].max()) / 2
    y = (corners[:, 1].min() + corners[:, 1].max()) / 2
    z = (corners[:, 2].min() + corners[:, 2].max()) / 2

    # 计算长宽高
    l = corners[:, 0].max() - corners[:, 0].min()
    w = corners[:, 1].max() - corners[:, 1].min()
    h = corners[:, 2].max() - corners[:, 2].min()

    return [float(x), float(y), float(z), float(l), float(w), float(h)]


def calculate_iou_3d(bbox1, bbox2):
    """
    计算两个3D bbox之间的IOU（Intersection over Union）
    bbox1, bbox2: 分别为两个3D bbox的坐标，每个bbox为一个数组，形状为 (8, 3)，包含八个角点的三维坐标
    返回IOU值
    """
    # 计算两个bbox的立方体体积
    def bbox_volume(bbox):
        min_coords = np.min(bbox, axis=0)
        max_coords = np.max(bbox, axis=0)
        side_lengths = max_coords - min_coords
        volume = np.prod(side_lengths)
        return volume

    # 提取bbox1和bbox2的坐标
    bbox1_min = np.min(bbox1, axis=0)
    bbox1_max = np.max(bbox1, axis=0)
    bbox2_min = np.min(bbox2, axis=0)
    bbox2_max = np.max(bbox2, axis=0)

    # 计算交集的立方体体积
    inter_min = np.maximum(bbox1_min, bbox2_min)
    inter_max = np.minimum(bbox1_max, bbox2_max)
    inter_side_lengths = inter_max - inter_min
    inter_volume = np.prod(np.maximum(inter_side_lengths, 0))

    # 计算并返回IOU
    bbox1_volume = bbox_volume(bbox1)
    bbox2_volume = bbox_volume(bbox2)
    iou = inter_volume / (bbox1_volume + bbox2_volume - inter_volume)
    return iou


def del_iOu(iou_threshold, prob, bbox):
    """
    删除IOU超过指定阈值的重叠框，保留概率最大的bbox
    iou_threshold: IOU阈值，超过该阈值的重叠框将保留概率较大的bbox
    prob: 每个bbox的概率值，形状为 (n,)
    bbox: 每个bbox的坐标，形状为 (n, 8, 3)，每个bbox包含八个角点的三维坐标
    返回保留的bbox
    """
    batch = bbox.shape[0]
    n = bbox.shape[1]
    keep_indices = -1*np.ones((batch,n))  # 初始化保留的bbox索引为所有索引

    # 计算每对bbox之间的IOU，并根据IOU阈值确定需要保留的bbox索引
    for b in range(bbox.shape[0]):
        for i in range(n):
            for j in range(i + 1, n):  # 只计算 i 和 j 之间的IOU，避免重复计算
                iou = calculate_iou_3d(bbox[b][i], bbox[b][j])
                if iou > iou_threshold:
                    # 根据概率值选择保留的bbox索引
                    if prob[b][i] > prob[b][j]:
                        keep_indices[b][j] = i  # 保留概率较大的bbox索引
                    else:
                        keep_indices[b][i] = j

    # 去除重复的保留索引，保留概率最大的bbox索引
    final_bbox = np.zeros((batch,1024,8,3))
    final_bbox_mask = np.zeros((batch,1024))
    for idx in range(len(keep_indices)):
        unique_keep_indices = np.unique(keep_indices[idx])
        unique_keep_indices = unique_keep_indices[unique_keep_indices != -1].astype('int64')
        for uni in range(len(unique_keep_indices)):
            final_bbox[idx][uni] = bbox[idx][unique_keep_indices[uni]]
            final_bbox_mask[idx][uni] = 1

    # # 根据保留的bbox索引选择最终保留的bbox
    # if len(unique_keep_indices) > 0:
    #     final_bbox = bbox[unique_keep_indices]
    # else:
    #     final_bbox = np.empty((0, 8, 3))

    return final_bbox,final_bbox_mask

# @torch.no_grad()
# def evaluate(
#         args,
#         curr_epoch,
#         model,
#         criterion,
#         dataset_config,
#         dataset_loader,
#         curr_train_iter,
# ):
#     # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
#     ap_calculator = APCalculator(
#         dataset_config=dataset_config,
#         ap_iou_thresh=[0.25, 0.5],
#         class2type_map=dataset_config.class2type,
#         no_nms=args.test_no_nms,
#         args=args
#     )
#
#     curr_iter = 0
#     net_device = next(model.parameters()).device
#     num_batches = len(dataset_loader)
#
#     loss_avg = SmoothedValue(window_size=10)
#     model.eval()
#     barrier()
#     epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
#
#     for batch_idx, batch_data_label in enumerate(dataset_loader):
#         batch_data_label = batch_dict_to_cuda(batch_data_label, local_rank=net_device)
#
#         inputs = {
#             "point_clouds": batch_data_label["point_clouds"],
#             "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
#             "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
#         }
#
#         outputs = model(inputs)
#         output = outputs['outputs']
#         #概率阈值
#         iou = 0.25
#         prob_threshold = 0
#         cla_threshold = 0
#
#         prob_mask = [output['objectness_prob']>prob_threshold][0].cpu().detach().numpy()
#
#
#         cls_prob = output['sem_cls_prob'].cpu().detach().numpy()
#         max_cls_ind = np.argmax(cls_prob, axis=2)
#
#         max_cls_scores = -10 * np.ones((cls_prob.shape[0], cls_prob.shape[1]))
#         # 遍历每个batch和候选框
#         for batch_idx in range(cls_prob.shape[0]):
#             for box_idx in range(cls_prob.shape[1]):
#                 # 获取当前候选框的最大类别索引
#                 max_cls_index = max_cls_ind[batch_idx, box_idx]
#
#                 # 根据最大类别索引,从分类结果矩阵中取出对应的得分
#                 max_cls_scores[batch_idx, box_idx] = cls_prob[batch_idx, box_idx, max_cls_index]
#
#
#         cls_mask = max_cls_scores>cla_threshold
#
#         prob_cls_mask = prob_mask&cls_mask
#         prob = output['objectness_prob'].cpu().detach().numpy()*(prob_cls_mask)
#
#         bbox = output['box_corners'].cpu().detach().numpy()*prob_cls_mask[:,:,None,None]
#         bbox = trans(bbox)
#
#         #删除重叠框
#         current_time = datetime.datetime.now()
#         file_name = f"/media/kou/Data1/htc/Chat-3D-v2/results/results_{current_time.strftime('%d_%H%M%S')}.json"
#         for b in range(len(bbox)):
#             result = {}
#             result['prob'] = list(prob[b].astype(float))
#             box = []
#             for i in range(len(bbox[b])):
#                 box.append(bbox_corners_to_xywlwh(bbox[b][i]))
#             result['bbox'] = box
#             with open(file_name,'a') as f:
#                 f.write(json.dumps({str(batch_data_label['scan_idx'][b].cpu().detach().numpy()): result}) + "\n")
#                 f.flush()
#         pro_bbox,pro_bbox_mask = del_iOu(iou,prob,bbox)
#
#         #保存结果
#         # 获取当前时间
#         current_time = datetime.datetime.now()
#         # 构建文件名
#         file_name = f"/media/kou/Data1/htc/Chat-3D-v2/results/results_{current_time.strftime('%d_%H%M%S')}.json"
#         for b in range(len(pro_bbox_mask)):
#             results = []
#             for i in range(int(np.sum(pro_bbox_mask[b]))):
#                 box = bbox_corners_to_xywlwh(pro_bbox[b][i])
#                 results.append(box)
#             with open(file_name,'a') as f:
#                 f.write(json.dumps({str(batch_data_label['scan_idx'][b].cpu().detach().numpy()): results}) + "\n")
#                 f.flush()
#         return 0,1,2


@torch.no_grad()
def evaluate(
        args,
        curr_epoch,
        model,
        criterion,
        dataset_config,
        dataset_loader,
        curr_train_iter,
):
    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        no_nms=args.test_no_nms,
        args=args
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        batch_data_label = batch_dict_to_cuda(batch_data_label, local_rank=net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }

        outputs = model(inputs)
        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)
            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"
        else:
            loss_dict_reduced = None

        if args.cls_loss.split('_')[0] == "focalloss":
            outputs["outputs"]["sem_cls_prob"] = outputs["outputs"]["sem_cls_prob"].sigmoid()

        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        if args.axis_align_test:
            outputs["outputs"]["box_corners"] = outputs["outputs"]["box_corners_axis_align"]

        ap_calculator.step_meter(outputs, batch_data_label)
        if is_primary() and curr_iter % args.log_every == 0:
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]"
            )
        curr_iter += 1
        barrier()

    return ap_calculator, loss_avg.avg, loss_dict_reduced