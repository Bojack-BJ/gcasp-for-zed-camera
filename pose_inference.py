import os
import sys
import torch
import numpy as np
import argparse
from eval import inference, load_model, get_args, infer_from_points
from NOCS_tools.utils import draw_detections, backproject
import DualSDFHandler

from mmdet.apis import inference_detector, init_detector
from mmdet.registry import VISUALIZERS
from mmengine.model import revert_sync_batchnorm
import cv2
import open3d as o3d

import pyzed.sl as sl

def init_model_ins(config, checkpoint, device='cuda:0'):
    model_ins = init_detector(config, checkpoint, device=device)
    if device == 'cpu':
        model_ins = revert_sync_batchnorm(model_ins)

    ins_visualizer = VISUALIZERS.build(model_ins.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    ins_visualizer.dataset_meta = model_ins.dataset_meta

    print("Instance Model initialized")
    return model_ins




# Parse arguments for pose estimation
parser = argparse.ArgumentParser( description='eval nocs')
parser.add_argument('config', type=str,help='The configuration file.')
parser.add_argument('--pretrained', default=None, type=str,help='pretrained model checkpoint')
parser.add_argument('--data', type=str, help="val/real_test", default='real_test')
parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")

args = parser.parse_args()
data = args.data


# Configurations for instance segmentation
ins_seg_config = '/home/lxt/mmdetection/configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py'
ins_seg_checkpoint = '/home/lxt/mmdetection/ckpt/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'

torch.multiprocessing.set_start_method('spawn')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')
cfg = get_args(args)

model_ins_seg = init_model_ins(ins_seg_config, ins_seg_checkpoint, device)



# Configurations for zed camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  
init_params.depth_mode = sl.DEPTH_MODE.NEURAL       
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_fps = 15                          
init_params.sdk_verbose = 1

tracking_parameters = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_parameters)

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

image = sl.Mat()
depth = sl.Mat()
scene_point_cloud = sl.Mat()
# sensors_data = sl.SensorsData()

if zed.grab() == sl.ERROR_CODE.SUCCESS:
    # zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
    zed.retrieve_image(image, sl.VIEW.LEFT)          
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)   
    zed.retrieve_measure(scene_point_cloud, sl.MEASURE.XYZRGBA)  

calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

# imu_data = sensors_data.get_imu_data()
# pose_matrix = imu_data.get_pose().m

left_cam = calibration_params.left_cam
fx = left_cam.fx
fy = left_cam.fy
cx = left_cam.cx
cy = left_cam.cy
intrinsics = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

image_ocv = image.get_data()
image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2RGB)
colors = image_ocv[:,:,:3].reshape(-1, 3)/255.


depth_ocv = depth.get_data()
depth_ocv_normalized = cv2.normalize(depth_ocv, None, 0, 255, cv2.NORM_MINMAX)
depth_ocv_normalized = depth_ocv_normalized.astype(np.uint8)

scene_points_3d = scene_point_cloud.get_data()[:, :, :3].reshape(-1, 3)
valid_mask = ~np.isnan(scene_points_3d).any(axis=1)  # 生成一个布尔掩码，标记没有 NaN 的点
scene_points_3d = scene_points_3d[valid_mask]
colors = colors[valid_mask]


# TEST!!!
# image_ocv = cv2.imread('/home/lxt/research-assignment/gcasp/0000_color.png')
# colors = image_ocv[:,:,:3].reshape(-1, 3)/255.
# depth_ocv = cv2.imread('/home/lxt/research-assignment/gcasp/0000_depth.png', -1)
# fake_mask = np.ones_like(depth_ocv)
# scene_points_3d, _ = backproject(depth_ocv, intrinsics, fake_mask)
# valid_mask = (depth_ocv > 0).flatten()
# colors = colors[valid_mask]


#Instance Segmentation
result_ins = inference_detector(model_ins_seg, image_ocv)
masks = result_ins.pred_instances.masks.cpu().numpy()
labels = result_ins.pred_instances.labels.cpu().numpy()
scores = result_ins.pred_instances.scores.cpu().numpy()
bboxes = result_ins.pred_instances.bboxes.cpu().numpy()

score_threshold = 0.3
keep = scores > score_threshold

masks = masks[keep]
labels = labels[keep]
scores = scores[keep] 
bboxes = bboxes[keep]

objects = {}
objects['points_3d'] = []
objects['label'] = []
objects['color'] = []
for mask, label in zip(masks, labels):
    indices = np.where(mask == True)  # 获取掩码区域的像素索引
    points_3d = []
    color = []
    for v, u in zip(indices[0], indices[1]):
        color.append(image_ocv[v, u]/255)
        point3D = scene_point_cloud.get_data()[v, u,:3]
        points_3d.append(point3D)
    color = np.array(color)
    points_3d = np.array(points_3d)  
    objects['points_3d'].append(points_3d)
    objects['label'].append(label)
    objects['color'].append(color)


# Load experimental settings
num_segs = cfg.data.num_segs 

synset_names = ['BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ]

class_map = {
    'bottle': 'bottle',
    'bowl':'bowl',
    'cup':'mug', # or can ?
    'laptop': 'laptop',
}

to_eval_ids = [] # check which category to evaluate, default: all
shape_handlers = [] # load dualsdf pretrained models

for eval_name in cfg.data.names:
        for i,name in enumerate(synset_names):
            if(name == eval_name):
                to_eval_ids.append(i)
                break

for eval_name in cfg.data.names:
    for i,name in enumerate(synset_names):
        if(name == eval_name):
            shape_handlers.append(DualSDFHandler.get_instance({
                'config': f"/home/lxt/research-assignment/gcasp/config/dualsdf{num_segs}.yaml",
                'pretrained': f"/home/lxt/research-assignment/gcasp/DualSDF_ckpts/{eval_name}/{num_segs}/epoch_9999.pth"
            }))
            break

model = load_model(args, cfg)

result = {}
result['pred_class_ids'] = []
result['pred_bboxes'] = []
result['pred_RTs'] = []
result['pred_scales'] = []
result['pred_scores'] = []
result['class_ids'] = []
result['masks'] = []

# intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
for i, eval_id in enumerate(to_eval_ids):
    print(f"Start inference for {synset_names[eval_id]}")
    for j,label in enumerate(objects['label']):
        print(f"Object {model_ins_seg.dataset_meta['classes'][label]}")
        if model_ins_seg.dataset_meta['classes'][label] == synset_names[eval_id] or (model_ins_seg.dataset_meta['classes'][label] == 'cup' and synset_names[eval_id] == 'mug'):

            valid_mask = ~np.isnan(objects['points_3d'][j]).any(axis=1)
            objects['points_3d'][j] = objects['points_3d'][j][valid_mask]


            clean_pc = o3d.geometry.PointCloud()
            clean_pc.points = o3d.utility.Vector3dVector(objects['points_3d'][j])
            clean_pc = clean_pc.voxel_down_sample(voxel_size=0.01)
            clean_pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

            points = np.asarray(clean_pc.points)

            s, R, t, T, pred_scale,shapecode, points = infer_from_points(model, points, shape_handlers[i], eval_id, num_segs)
            if points is None:
                continue
            o3d_pc = o3d.geometry.PointCloud()
            # s, R, t, T, pred_scale,shapecode, points = infer_from_points(model, depth_ocv, masks[j], None, intrinsics, shape_handlers[i], eval_id)
            object_colors = np.zeros((points.shape[0], 3))
            object_colors[:, 0] = 1.0
            object_center = np.mean(points, axis=0)
            # points = np.vstack((scene_points_3d, points))
            # object_colors = np.vstack((colors, object_colors))
            o3d_pc.points = o3d.utility.Vector3dVector(scene_points_3d)
            o3d_pc.colors = o3d.utility.Vector3dVector(colors)
            o3d_pc = o3d_pc.voxel_down_sample(voxel_size=0.01)
            o3d_pc, ind = o3d_pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # object_center = np.mean(objects['points_3d'][j], axis=0)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            z_180_RT = np.zeros((4, 4), dtype=np.float32)
            z_180_RT[:3, :3] = np.diag([-1, -1, 1])
            z_180_RT[3, 3] = 1
            axis.rotate(z_180_RT[:3, :3])
            axis.rotate(T[:3,:3])
            axis.translate(object_center)
            o3d.visualization.draw_geometries([o3d_pc, axis])
            normalized_object_pc = o3d.geometry.PointCloud()
            normalized_object_pc.points = o3d.utility.Vector3dVector(objects['points_3d'][j])
            normalized_object_pc = normalized_object_pc.rotate(np.linalg.inv(T[:3,:3]))
            normalized_object_pc = normalized_object_pc.translate(-T[:3,3])
            normalized_object_pc = normalized_object_pc.rotate(z_180_RT[:3, :3])
            normalized_object_pc.colors = o3d.utility.Vector3dVector(object_colors)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            o3d.visualization.draw_geometries([normalized_object_pc, axis])
            o3d.io.write_point_cloud(f"/home/lxt/research-assignment/PoinTr/data/{synset_names[eval_id]}_{j}.pcd", normalized_object_pc)
            print(f"Object {synset_names[eval_id]}: ")
            # print(f"s: {s}")
            # print(f"R: {R}")
            # print(f"t: {t}")
            # print(f"object_center: {object_center}")
            # print(f"T: {T}")
            # print(f"pred_scale: {pred_scale}")
            # print(f"shapecode: {shapecode}")


zed.close()
# draw_detections(image_ocv, './', data, 'image', intrinsics, synset_names, True,
#                                     0, 0, 0, 0, 0, 0, 0,
#                                     result['pred_bboxes'], result['class_ids'], result['masks'], 0, np.array(result['pred_RTs']), result['pred_scores'], np.array(result['pred_scales']))