import argparse
import torch
import torchvision.transforms as transforms
import pathlib
import os
import json

from Panoptic_configs.config import config
from Panoptic_configs.config import update_config

from Panoptic_dataset_class.panoptic import Panoptic
from Panoptic_dataset_class.utils.vis import save_annotations_vis_img

'''
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
'''
    
def parse_args():
	parser = argparse.ArgumentParser(
		description='Yolo-x model train/inference')
	
	# Path
	parser.add_argument('--panoptic_config_file_path', help='Panoptic config file path', default='./Panoptic_configs/Panoptic_annotations_builder_config.yaml')	
	
	args = parser.parse_args()
	
	return args

def main():
	# Args parsing
	args = parse_args()
	
	# Update config
	update_config(args.panoptic_config_file_path)

	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		
	gpus = [int(i) for i in config.GPUS.split(',')]
	ANNOTATION_BUILDER_LIST = config.DATASET.SEQ_LIST
	cam_list = config.DATASET.CAMERA_LIST
	
	for seq in ANNOTATION_BUILDER_LIST:
		# Load Panoptic
		panoptic_dataset = Panoptic( \
			config, config.DATASET.TEST_SUBSET, False, seq, \
			cam_list = cam_list, \
			interval = 1, \
			transform = transforms.Compose([
				transforms.ToTensor(),
				normalize,
				])
			)

		# Batch loader
		
		loader = torch.utils.data.DataLoader(
			panoptic_dataset,
			batch_size=config.TEST.BATCH_SIZE * len(gpus),
			shuffle=False,
			num_workers=config.WORKERS,
			pin_memory=True)
			
		# Save data
		with torch.no_grad():
			for b, batch_data in enumerate(loader):
			
				inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap = batch_data
				
				for view_num, (meta, input) in enumerate(zip(meta, inputs)):
					
					# Annotation
					data = {}
					data['bodies'] = []

					# Path
					seq_name = meta['seq'][0]
					image_id = meta['image'][0].split('_')[-1][:-4]
					view_id = meta['view_id'][0]
					file_name = view_id+'_'+str(image_id)
					dir_path = os.path.join('.', config.OUTPUT_DIR, seq_name, view_id)

					# Make dir			
					pathlib.Path(os.path.join(dir_path, 'annotations')).mkdir(parents=True, exist_ok=True)	
					pathlib.Path(os.path.join(dir_path, 'images')).mkdir(parents=True, exist_ok=True) 
	    			
					for id_num in meta['id']:
						
						id_num = int(id_num.cpu().numpy())
						
						# Data list
						transformed_joints_3d = []
						transformed_joints_3d_vis = []
						projected_joints_2d = []
						projected_joints_2d_vis = []
						bb = []
						bb_clip = []
						bb_vis = []
						bb_vis_clip = []
						
						# 3D keypoints
						for (j, vis) in zip(meta['joints_3d'][0][id_num], meta['joints_3d_vis'][0][id_num]):
							transformed_joints_3d += [float(j[0].cpu().numpy()), float(j[1].cpu().numpy()), float(j[2].cpu().numpy())]
							transformed_joints_3d_vis += [float(vis[0].cpu().numpy())]
							
						# 2D keypoints
						for (j, vid) in zip(meta['joints'][0][id_num], meta['joints_vis'][0][id_num]):
							projected_joints_2d += [float(j[0].cpu().numpy()), float(j[1].cpu().numpy())]
							projected_joints_2d_vis += [float(vis[0].cpu().numpy())]							
						
						# 2D bounding boxes
						box = meta['bounding_boxes'][0][id_num]
						bb += [float(box[0].cpu().numpy()), float(box[1].cpu().numpy()), float(box[2].cpu().numpy()), float(box[3].cpu().numpy())]

						box_clip = meta['bounding_boxes_clip'][0][id_num]
						bb_clip += [float(box_clip[0].cpu().numpy()), float(box_clip[1].cpu().numpy()), float(box_clip[2].cpu().numpy()), float(box_clip[3].cpu().numpy())]

						box_vis = meta['bounding_boxes_vis'][0][id_num]
						bb_vis += [float(box_vis[0].cpu().numpy()), float(box_vis[1].cpu().numpy()), float(box_vis[2].cpu().numpy()), float(box_vis[3].cpu().numpy())]

						box_vis_clip = meta['bounding_boxes_vis_clip'][0][id_num]
						bb_vis_clip += [float(box_vis_clip[0].cpu().numpy()), float(box_vis_clip[1].cpu().numpy()), float(box_vis_clip[2].cpu().numpy()), float(box_vis_clip[3].cpu().numpy())]
						
						data['bodies'].append({
							"view_id": view_id,
							"id": int(id_num),
							"num_person": int(meta['num_person'].cpu().numpy()),
							"input_width": input.cpu().numpy().shape[-1],
							"input_height": input.cpu().numpy().shape[-2],
							"transformed_joints_3d": transformed_joints_3d,
							"transformed_joints_3d_vis": transformed_joints_3d_vis,
							"projected_joints_2d": projected_joints_2d,
							"projected_joints_2d_vis": projected_joints_2d_vis,
							"bbox": bb,
							"bbox_clip": bb_clip,
							"bbox_vis": bb_vis,
							"bbox_vis_clip": bb_vis_clip
						})

					# Save annotations
					with open(os.path.join(dir_path, 'annotations', '{}_gt.json'.format(file_name)), 'w') as outfile:
						json.dump(data, outfile)
					#print('Save anno file:', os.path.join(dir_path, 'annotations', '{}_gt.json'.format(file_name)))

					# Save GT visualization img
					save_annotations_vis_img(dir_path, file_name, input, meta) 
			
				if b % 100 == 0:
					print(f"{seq}: {meta['seq'][0]}, idx: {b}, image_id: {meta['image'][0]}")

if __name__ == '__main__':
	main()
