# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import os
import copy

from .JointsDataset import JointsDataset
from .utils.transforms import projectPoints

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    '160906_band3',
]

VAL_LIST = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4']

ANNOTATION_BUILDER_LIST = [
    '160422_ultimatum1',
    '160224_haggling1',
    '160226_haggling1',
    '161202_haggling1',
    '160906_ian1',
    '160906_ian2',
    '160906_ian3',
    '160906_band1',
    '160906_band2',
    '160906_band3',
    '160906_pizza1',
    '160422_haggling1',
    '160906_ian5',
    '160906_band4'
    ]

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    'l-eye': 15,
    'l-ear': 16,
    'r-eye': 17,
    'r-ear': 18
}

LIMBS = [[0, 1],
         [0, 2],
         [0, 3],
         [3, 4],
         [4, 5],
         [0, 9],
         [9, 10],
         [10, 11],
         [2, 6],
         [2, 12],
         [6, 7],
         [7, 8],
         [12, 13],
         [13, 14]]

class Panoptic(JointsDataset):
    def __init__(self, cfg, image_set, is_train, sequence, cam_list=None, interval=1, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.pixel_std = 200.0
        self.joints_def = JOINTS_DEF
        self.limbs = LIMBS
        self.num_joints = len(JOINTS_DEF)
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)

        if self.image_set == 'train':
            self.sequence_list = TRAIN_LIST
            self._interval = 3
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            # self.cam_list = list(set([(0, n) for n in range(0, 31)]) - {(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)})
            # self.cam_list.sort()
            self.num_views = len(self.cam_list)
        elif self.image_set == 'validation':
            self.sequence_list = VAL_LIST
            self._interval = 12
            self.cam_list = [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)][:self.num_views]
            self.num_views = len(self.cam_list)
        elif self.image_set == 'annotation_builder':
            self.sequence_list = [sequence] #ANNOTATION_BUILDER_LIST
            self._interval = interval
            self.cam_list = [(0, cam) for cam in cam_list] #[(0,i) for i in range(31)][:self.num_views]
            self.num_views = len(self.cam_list)
        
        print('Sequence list:', self.sequence_list)
        print('Camera list:', self.cam_list)
        
        self.db_file = 'group_{}_cam{}.pkl'.format(self.image_set, self.num_views)
        self.db_file = os.path.join(self.dataset_root, self.db_file)
        
        print('DB file path:', self.db_file)
       
        self.db = self._get_db()
        print('--- Complete to build Panoptic DB! ---')
        
        self.db_size = len(self.db)
        print('DB size:', self.db_size)

    def _get_human_bb_from_pose2d(self, pose2d, joints_vis, width=1920, height=1080, \
        width_offset_div=5, height_offset_div=5, points_thr=0):
        
        boxes = []
        vis_pose2d = []
        # offset_height = height #self.image_size[0]
        # offset_width = width #self.image_size[1]
        # offset = (offset_width / width_offset_div, offset_height / up_offset_div, offset_height / under_offset_div)
        
        for (p, v) in zip(pose2d, joints_vis):
            
            if v:
                vis_pose2d.append(p)
        len_vis = len(vis_pose2d)

        if len_vis > points_thr:
            
            vis_pose2d = np.array(vis_pose2d)
            target_pose_list = [pose2d, vis_pose2d]

            for target_pose in target_pose_list:
                # Max, min axis
                x_min = np.min(target_pose[:,0])
                x_max = np.max(target_pose[:,0])
                y_min = np.min(target_pose[:,1])
                y_max = np.max(target_pose[:,1])
                
                # Index of max, min axis
                # index_x_min = np.argmin(target_pose[:,0])
                # index_x_max = np.argmax(target_pose[:,0])
                # index_y_min = np.argmin(target_pose[:,1])
                # index_y_max = np.argmax(target_pose[:,1])

                # Box width, height
                bb_width = float(x_max - x_min)
                bb_height = float(y_max - y_min)
                width_ratio = 0
                height_ratio = 0

                # Set offset
                if bb_width != 0:

                    num_distance = 0
                    left_eye_nose_distance = 0
                    if joints_vis[1] and joints_vis[15]:
                        left_eye_nose_distance = np.linalg.norm(np.array(pose2d[1]) - np.array(pose2d[15]))
                        num_distance += 1
                    right_eye_nose_distance = 0
                    if joints_vis[1] and joints_vis[17]:
                        right_eye_nose_distance = np.linalg.norm(np.array(pose2d[1]) - np.array(pose2d[17]))
                        num_distance += 1

                    if num_distance != 0:
                        eye_nose_distance = (left_eye_nose_distance + right_eye_nose_distance) / float(num_distance)
                        offsets = [eye_nose_distance * 3.5, eye_nose_distance * 3.5, eye_nose_distance * 3, eye_nose_distance * 4]
                    else: 
                        # width_ratio = bb_width / width
                        height_ratio = bb_height / height

                        offsets = [(height / width_offset_div) * (height_ratio / 2), (height / width_offset_div) * (height_ratio / 2), (height / height_offset_div) * (height_ratio / 2), (height / height_offset_div) * (height_ratio / 2)]

                    # if index_x_min == 3:
                    #     offsets[0] = offsets[0] / 2
                    # elif index_x_min == 4:
                    #     offsets[0] = offsets[0] / 2
                    
                    # if index_x_max == 9:
                    #     offsets[1] = offsets[1] / 2
                    # elif index_x_max == 10:
                    #     offsets[1] = offsets[1] / 2

                else:
                    offsets = [100, 100, 100, 100]

                x_lef_top = x_min - offsets[0]
                y_lef_top = y_min - offsets[2]
                x_rig_bot = x_max + offsets[1]
                y_rig_bot = y_max + offsets[3]
                x_lef_top_clip = np.clip(x_min - offsets[0], 0, width-1)
                y_lef_top_clip = np.clip(y_min - offsets[2], 0, height-1)
                x_rig_bot_clip = np.clip(x_max + offsets[1], 0, width-1)
                y_rig_bot_clip = np.clip(y_max + offsets[3], 0, height-1)	 
                
                boxes.append([x_lef_top, y_lef_top, x_rig_bot, y_rig_bot])
                boxes.append([x_lef_top_clip, y_lef_top_clip, x_rig_bot_clip, y_rig_bot_clip])
        
        else:
            for i in range(4):
                boxes.append([0, 0, 0, 0])
                
        return boxes

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        
        print('--- Build Panoptic DB! ---')
        
        for seq in self.sequence_list:

            cameras = self._get_cam(seq)

            print('Sequence:', seq)
            
            curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
            self.anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

            print('Length of annotation files:', len(self.anno_files))

            for i, file in enumerate(self.anno_files):
                if i % self._interval == 0:
                    with open(file) as dfile:
                        try:
                        	bodies = json.load(dfile)['bodies']
                        except Exception as e:
                        	bodies = []
                    
                    for k, v in cameras.items():
                        postfix = osp.basename(file).replace('body3DScene', '')
                        prefix = '{:02d}_{:02d}'.format(k[0], k[1])
                        image = osp.join(seq, 'hdImgs', prefix,
                                         prefix + postfix)
                        image = image.replace('json', 'jpg')

                        all_ids = []
                        all_poses_3d = []
                        all_poses_vis_3d = []
                        all_poses = []
                        all_poses_vis = []
                        all_bbs = []
                        all_bbs_clip = []
                        all_vis_bbs = []
                        all_vis_bbs_clip = []
                        
                        for id_num, body in enumerate(bodies):
                            pose3d = np.array(body['joints19']).reshape((-1, 4))
                            pose3d = pose3d[:self.num_joints]

                            joints_vis = pose3d[:, -1] > 0.1

                            if not joints_vis[self.root_id]:
                                continue

                            # Coordinate transformation
                            M = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 0.0, -1.0],
                                          [0.0, 1.0, 0.0]])
                            pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

                            all_poses_3d.append(pose3d[:, 0:3] * 10.0)
                            all_poses_vis_3d.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 3, axis=1))

                            pose2d = np.zeros((pose3d.shape[0], 2))
                            pose2d[:, :2] = projectPoints(
                                pose3d[:, 0:3].transpose(), v['K'], v['R'],
                                v['t'], v['distCoef']).transpose()[:, :2]
                                
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                                     pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0
                            
                            boxes = self._get_human_bb_from_pose2d(pose2d, joints_vis, width, height)

                            all_ids.append(id_num)
                            all_poses.append(pose2d)
                            all_poses_vis.append(
                                np.repeat(
                                    np.reshape(joints_vis, (-1, 1)), 2, axis=1))
                            all_bbs.append(boxes[0])
                            all_bbs_clip.append(boxes[1])
                            all_vis_bbs.append(boxes[2])
                            all_vis_bbs_clip.append(boxes[3])                            
                            
                        #if len(all_poses_3d) > 0:
                        our_cam = {}
                        our_cam['R'] = v['R']
                        our_cam['T'] = -np.dot(v['R'].T, v['t']) * 10.0  # cm to mm
                        our_cam['fx'] = np.array(v['K'][0, 0])
                        our_cam['fy'] = np.array(v['K'][1, 1])
                        our_cam['cx'] = np.array(v['K'][0, 2])
                        our_cam['cy'] = np.array(v['K'][1, 2])
                        our_cam['k'] = v['distCoef'][[0, 1, 4]].reshape(3, 1)
                        our_cam['p'] = v['distCoef'][[2, 3]].reshape(2, 1)

                        db.append({
                            'key': "{}_{}{}".format(seq, prefix, postfix.split('.')[0]),
                            'view_id': prefix,
                            'seq': seq,
                            'image': osp.join(self.dataset_root, image),
                            'id': all_ids,
                            'joints_3d': all_poses_3d,
                            'joints_3d_vis': all_poses_vis_3d,
                            'joints_2d': all_poses,
                            'joints_2d_vis': all_poses_vis,
                            'bounding_boxes': all_bbs,
                            'bounding_boxes_clip': all_bbs_clip,
                            'vis_bounding_boxes': all_vis_bbs,
                            'vis_bounding_boxes_clip': all_vis_bbs_clip,
                            'camera': our_cam
                        })
        
        return db

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, 'calibration_{:s}.json'.format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib['cameras']:
            if (cam['panel'], cam['node']) in self.cam_list:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['distCoef'] = np.array(cam['distCoef'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((3, 1))
                cameras[(cam['panel'], cam['node'])] = sel_cam
        return cameras

    def __getitem__(self, idx):
        input, target, weight, target_3d, meta, input_heatmap = [], [], [], [], [], []

        # if self.image_set == 'train':
        #     # camera_num = np.random.choice([5], size=1)
        #     select_cam = np.random.choice(self.num_views, size=5, replace=False)
        # elif self.image_set == 'validation':
        #     select_cam = list(range(self.num_views))

        for k in range(self.num_views):
            i, t, w, t3, m, ih = super().__getitem__(self.num_views * idx + k)
            if i is None:
                continue
            input.append(i)
            target.append(t)
            weight.append(w)
            target_3d.append(t3)
            meta.append(m)
            input_heatmap.append(ih)
        return input, target, weight, target_3d, meta, input_heatmap

    def __len__(self):
        return self.db_size // self.num_views

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size // self.num_views
        assert len(preds) == gt_num, 'number mismatch'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_views * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']

            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        return aps, recs, self._eval_list_to_mpjpe(eval_list), self._eval_list_to_recall(eval_list, total_gt)

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt
