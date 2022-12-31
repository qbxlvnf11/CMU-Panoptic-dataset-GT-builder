# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import math
import numpy as np
import torchvision
import cv2
import os
import pathlib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
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
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}
'''

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
    
def save_batch_image_with_joints_bb_multi(batch_image,
                                 batch_joints,
                                 batch_joints_vis,
                                 bbs,
                                 bbs_clip,
                                 bbs_vis,
                                 bbs_vis_clip,
                                 num_person,
                                 ids,
                                 nrow,
                                 padding,
                                 file_name=None,
                                 save_flag=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_person, num_joints, 3],
    batch_joints_vis: [batch_size, num_person, num_joints, 1],
    num_person: [batch_size]
    }
    '''
    batch_image = batch_image.flip(1)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            for n in range(num_person[k]):
                
                # Bounding boxes
                id_num = n
                thickness = 1
                
                label = '{}{:d}'.format("", id_num)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
                
                # All
                box = bbs_clip[0][id_num]
                x1, y1, x2, y2 = [int(b.cpu().numpy()) for b in box] 
                color = (0, 0, 255)   
                cv2.rectangle(ndarr, (x1, y1), (x2, y2), color, thickness)
                cv2.rectangle(ndarr, (x1, y1), (x1-t_size[0]+3, y1+t_size[1]+4), color, -1)
                cv2.putText(ndarr, label, (x1-t_size[0]+3, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)

                # Vis
                box = bbs_vis_clip[0][id_num]
                x1, y1, x2, y2 = [int(b.cpu().numpy()) for b in box] 
                color = (0, 255, 0)
                cv2.rectangle(ndarr, (x1, y1), (x2, y2), color, thickness)
                cv2.rectangle(ndarr, (x1, y1), (x1-t_size[0]+3, y1+t_size[1]+4), color, -1)
                cv2.putText(ndarr, label, (x1-t_size[0]+3, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
                
                # 2D joints
                joints = batch_joints[k, n]
                joints_vis = batch_joints_vis[k, n]

                for joint, joint_vis in zip(joints, joints_vis):
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    if joint_vis[0]:
                        cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2,
                                   [0, 255, 255], 2)
                                   
                for i, skel in enumerate(LIMBS):
                    
                    pt1_idx = skel[0]
                    pt2_idx = skel[1]
                    
                    pt1_x = joints[pt1_idx][0]
                    pt1_y = joints[pt1_idx][1]
                    pt2_x = joints[pt2_idx][0]
                    pt2_y = joints[pt2_idx][1]
                    
                    if joints_vis[pt1_idx][0] and joints_vis[pt2_idx][0]:
                        frame = cv2.line(
                                   ndarr, (int(pt1_x), int(pt1_y)), (int(pt2_x), int(pt2_y)),
                                   [0, 255, 255], 1
                        )
           
            k = k + 1
    
    if save_flag:
        cv2.imwrite(file_name, ndarr)
    else:
        return ndarr

def save_batch_heatmaps_multi(batch_image, batch_heatmaps, file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)
    batch_image = batch_image.flip(1)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8)

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)

def save_annotations_vis_img(dir_path, file_name, input, meta, nrow=8, padding=2):	
    
    # Draw anno
    vis_img = save_batch_image_with_joints_bb_multi(input, meta['joints'], meta['joints_vis'], \
        meta['bounding_boxes'], meta['bounding_boxes_clip'], \
        meta['vis_bounding_boxes'], meta['vis_bounding_boxes_clip'], \
        meta['num_person'], meta['id'], save_flag=False, nrow=nrow, padding=padding)
    
    # Save image
    cv2.imwrite(os.path.join(dir_path, 'vis_images', '{}_vis.jpg'.format(file_name)), vis_img)

def save_origin_img(dir_path, file_name, batch_image, nrow=8, padding=2):	

    batch_image = batch_image.flip(1)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    
    # Save image
    cv2.imwrite(os.path.join(dir_path, 'origin_images', '{}.jpg'.format(file_name)), ndarr)

def save_debug_3d_images(config, meta, preds, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, '3d_joints')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_3d.png"

    # preds = preds.cpu().numpy()
    batch_size = meta['num_person'].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        num_person = meta['num_person'][i]
        joints_3d = meta['joints_3d'][i]
        joints_3d_vis = meta['joints_3d_vis'][i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')
        for n in range(num_person):
            joint = joints_3d[n]
            joint_vis = joints_3d_vis[n]
            for k in eval("LIMBS{}".format(len(joint))):
                if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)
                else:
                    x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                    y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                    z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                    ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                            markeredgewidth=1)

        colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
        if preds is not None:
            pred = preds[i]
            for n in range(len(pred)):
                joint = pred[n]
                if joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
    plt.savefig(file_name)
    plt.close(0)


def save_debug_3d_cubes(config, meta, root, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, 'root_cubes')

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_root.png"

    batch_size = root.shape[0]
    root_id = config.DATASET.ROOTIDX

    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 6.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.95, wspace=0.05, hspace=0.15)
    for i in range(batch_size):
        roots_gt = meta['roots_3d'][i]
        num_person = meta['num_person'][i]
        roots_pred = root[i]
        ax = plt.subplot(yplot, xplot, i + 1, projection='3d')

        x = roots_gt[:num_person, 0].cpu()
        y = roots_gt[:num_person, 1].cpu()
        z = roots_gt[:num_person, 2].cpu()
        ax.scatter(x, y, z, c='r')

        index = roots_pred[:, 3] >= 0
        x = roots_pred[index, 0].cpu()
        y = roots_pred[index, 1].cpu()
        z = roots_pred[index, 2].cpu()
        ax.scatter(x, y, z, c='b')

        space_size = config.MULTI_PERSON.SPACE_SIZE
        space_center = config.MULTI_PERSON.SPACE_CENTER
        ax.set_xlim(space_center[0] - space_size[0] / 2, space_center[0] + space_size[0] / 2)
        ax.set_ylim(space_center[1] - space_size[1] / 2, space_center[1] + space_size[1] / 2)
        ax.set_zlim(space_center[2] - space_size[2] / 2, space_center[2] + space_size[2] / 2)

    plt.savefig(file_name)
    plt.close(0)
