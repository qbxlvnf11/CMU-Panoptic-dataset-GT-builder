CMU Panoptic Dataset
=============

#### - Details of CMU Panoptic dataset
  - Capture the 3D motion of a group of people engaged in a social interaction
  - The studio structure
  
  <img src="https://user-images.githubusercontent.com/52263269/202191388-473de652-5c21-40cc-9218-96dc939e9724.png" width="50%"></img>

  - Massively multiview system
    - Hardware-based synchronized 480 VGA cameras views
      - 640 x 480 resolution, 25 fps
    - Hardware-based synchronized 31 HD cameras views
      - 1920 x 1080 resolution, 30 fps
    - Camera calibration
    - 10 RGB-D sensors (10 Kinect Ⅱ Sensors)
      - 1920 x 1080 (RGB), 512 x 424 (depth), 30 fps
      - Synchronized with HD cameras
    
  - Multiple people
    - 3D body pose
    - 3D facial landmarks
    
  - Data list examples
    - VoxelPose: 160422_ultimatum1, 160224_haggling1, 160226_haggling1 etc.
    
#### - CMU Panoptic dataset structure

```
|-- 160224_haggling1
            |   |-- hdImgs
            |   |-- hdvideos
            |   |-- hdPose3d_stage1_coco19
            |   |-- calibration_160224_haggling1.json
|-- 160226_haggling1  
            |-- ...
```

Generated Annotations
=============

#### - Generating annotation json files and visualized images

  <img src="https://user-images.githubusercontent.com/52263269/202971006-dd8733de-a8d5-471c-bdb9-20702657e6e1.jpg" width="40%"></img>
  <img src="https://user-images.githubusercontent.com/52263269/202972107-75075843-87d2-4aa5-8b23-c1dbc835cac7.jpg" width="40%"></img>
  <img src="https://user-images.githubusercontent.com/52263269/202972266-2bf65cf9-bd80-47ea-a714-f8bf0d396e80.jpg" width="40%"></img>
  <img src="https://user-images.githubusercontent.com/52263269/202972373-875bcedd-3ac3-42ed-8e7a-2ac8de02dc3d.jpg" width="40%"></img>

#### - Annotation structures

```
|-- 160224_haggling1
            |   |-- calibration_160422_haggling1.json
            |   |-- 00_01
            |   |   |   |-- annotations
            |   |   |   |   |   |-- 00_03_00000206_gt.json
            |   |   |   |   |   |-- ...
            |   |   |   |-- origin_images
            |   |   |   |   |   |-- 00_03_00000206.jpg
            |   |   |   |   |   |-- ...
            |   |   |   |-- vis_images
            |   |   |   |   |   |-- 00_03_00000206_vis.jpg
            |   |   |   |   |   |-- ...
            |   |-- 00_02
            |   |-- ...
|-- 160226_haggling1  
            |-- ...
```

#### - Annotation json file format
  
```
{"bodies": [
  { "view_id": view id (HD camera id),
  "id": person id,
  "num_person": number of the people,
  "input_width": image width (1920),
  "input_height": image height (1080),
  "transformed_joints_3d": GT transformed joints 3d,
  "transformed_joints_3d_vis": visualization flags of joints 3d,
  "projected_joints_2d": GT joints 2d projected by joints 3d using camera parameters in each view,
  "projected_joints_2d_vis": visualization flags of joints 2d,
  "bbox": bounding boxes created by adding and subtracting an offset from the min/maxvalues of x and y values of each person's GT 2D keypoint,
  "bbox_clip": bbox cliped by image size,
  "vis_bbox": bounding boxes created by adding and subtracting an offset from the min/max values of x and y values of each person's GT 2D keypoint that visualization flag value is true,
  "vis_bbox_clip": vis_bbox cliped by image size }
  , ...
  ]
}
```

#### - Keypoints format
  
```
0: Neck
1: Nose
2: BodyCenter (center of hips)
3: lShoulder
4: lElbow
5: lWrist,
6: lHip
7: lKnee
8: lAnkle
9: rShoulder
10: rElbow
11: rWrist
12: rHip
13: rKnee
14: rAnkle
15: lEye
16: lEar
17: rEye
18: rEar
```

  - 3d keypoints: [x0, y0, z0, x1, y1, z1, ...]
  - 2d keypoints: [x0, y0, x1, y1, ...]

#### - Bounding box format of each 2d view
  - Box format: [left_top_x, left_top_y, right_bottom_x, right_bottom_y]
  - A box of people that has 3d coordinates but is not visible in the 2d view has coordinates [0, 0, 0, 0]

Docker Environments
=============

#### - Pull docker environment

```
docker pull qbxlvnf11docker/panoptic_dataset_env:latest
```

#### - Run docker environment

```
nvidia-docker run -it -p 9000:9000 -e GRANT_SUDO=yes --user root --name panoptic_dataset_env --shm-size=4G -v {folder}:/workspace -w /workspace qbxlvnf11docker/panoptic_dataset_env bash
```

How to use
=============

#### - Building Panoptic dataset annotations
  - Select the dataset and camera id to extract annotations by editing config file

```
python main.py --panoptic_config_file_path ./Panoptic_configs/Panoptic_annotations_builder_config.yaml
```

#### - Panoptic dataset download & preparation
  - Variable 'datasets': select the sequences to download
  - Variable 'nodes': select the camera ids to download

```
apt-get install wget
cd ./Panoptic_download_toolbox_scripts
./getData_list.sh
./extractAll_list.sh
```

References
=============

#### - CMU Panoptic dataset paper
```
@article{CMU Panoptic Dataset,
  title={Panoptic Studio: A Massively Multiview System for Social Interaction Capture},
  author={Hanbyul Joo et al.},
  journal = {arXiv},
  year={2016}
}
```

#### - CMU Panoptic dataset

https://www.cs.cmu.edu/~hanbyulj/panoptic-studio/

https://paperswithcode.com/dataset/cmu-panoptic

#### - CMU Panoptic dataset download toolbox

https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox

#### - CMU Panoptic Pytorch dataset class 

https://github.com/microsoft/voxelpose-pytorch

Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com
