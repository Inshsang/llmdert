import os
import open3d as o3d


def bbox_xyzlwh_to_corners(bbox_xyzlwh):
    """ Convert a bounding box from xyzlwh format to corners format (8 points). """
    x, y, z, l, w, h = bbox_xyzlwh
    half_l = l / 2.0
    half_w = w / 2.0
    half_h = h / 2.0

    corners = [
        [x - half_l, y - half_h, z - half_w],  # front-left-bottom
        [x + half_l, y - half_h, z - half_w],  # front-right-bottom
        [x + half_l, y - half_h, z + half_w],  # front-right-top
        [x - half_l, y - half_h, z + half_w],  # front-left-top
        [x - half_l, y + half_h, z - half_w],  # back-left-bottom
        [x + half_l, y + half_h, z - half_w],  # back-right-bottom
        [x + half_l, y + half_h, z + half_w],  # back-right-top
        [x - half_l, y + half_h, z + half_w]  # back-left-top
    ]

    return corners

def showbbox(bbox,points):
    # 创建线框网格，连接 OBB 的八个顶点
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    # bbox = event.metadata['objects'][3]['objectOrientedBoundingBox']['cornerPoints']
    # assetid = event.metadata['objects'][3]['assetId']
    # print(assetid)
    # boundingbox = gt0['999']
    # boundingbox = boundingbox['Chair|2|1|1']['objectOrientedBoundingBox']['cornerPoints']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(points[:,3:])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # 添加点云、线框网格和标签到 visualizer
    vis.add_geometry(pcd)

    for box in bbox:
        # box = bbox_xyzlwh_to_corners(box[:6])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        vis.add_geometry(line_set)

    # o3d.io.write_point_cloud('/media/kou/Data1/htc/Chat-3D-v2/V-DETR/myscene/processed_scene.ply', pcd)
    # 查看可视化结果
    vis.run()
    vis.destroy_window()
    return 0