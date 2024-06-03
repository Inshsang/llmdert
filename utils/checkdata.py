import json
import numpy as np
import os
import jsonlines


def process_gt(gt,object_dict):
    instance_bboxes = []
    # 遍历输入的 gt
    for obj in gt:
        # 获取物体名称
        if not len(obj):
            continue
        obj_name = obj['name'].lower()
        # 查找物体名称在字典中的序号
        if obj_name in object_dict:
            obj_index = object_dict[obj_name]
            # 在对应位置设置为 1
            obj['BoundingBox'].append(obj_index)
            instance_bboxes.append(obj['BoundingBox'])

    return np.asarray(instance_bboxes)


class_mapping = {
    "cabinet": 0,  #
    "bed": 1,
    "chair": 2,
    "sofa": 3,
    "diningtable": 4,
    "doorway": 5,
    "window": 6,
    "shelf": 7,
    "painting": 8,
    "countertop": 9,
    "desk": 10,
    "curtain": 11,  #
    "fridge": 12,
    "showercurtrain": 13,  #
    "toilet": 14,
    "sink": 15,
    "bathtub": 16,  #
    "garbagecan": 17,
}
all_scan_names = list(
    set(
        [
            int(x[:-4])
            for x in os.listdir('/media/kou/Data3/htc/scene')
        ]
    )
)
all_scan_names.sort()
scan_names = all_scan_names[0:30000]
to_remove = []
gt = {}
GT = open('/media/kou/Data1/htc/MYDATA/BenchMark/Task/GT/Detection.json')
for item in jsonlines.Reader(GT):
    for key, value in item.items():
        gt[key] = value
gt = gt
# to_remove = [207, 278, 772, 808, 986, 1203, 1333, 1407, 1662, 2829, 3077, 3933, 4108, 4587, 4621, 6062, 6301, 6559,
#              6614, 6728, 7498, 8374, 8750, 9294, 9462, 9532, 9716]
# all_scan_names = [name for name in all_scan_names if name not in to_remove]
for i in scan_names:
    try:
        instance_bboxes = process_gt(gt[str(i)],class_mapping)
    except Exception as e:
        print(i)
        print(gt[str(i)])
    if instance_bboxes.shape == (0,):
        to_remove.append(i)
print(len(all_scan_names))
print(to_remove)