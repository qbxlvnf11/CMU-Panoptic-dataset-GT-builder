CMU Panoptic
=============

#### - CMU Panoptic dataset
  - Capture the 3D motion of a group of people engaged in a social interaction
  - The studio structure
  
  <img src="https://user-images.githubusercontent.com/52263269/202191388-473de652-5c21-40cc-9218-96dc939e9724.png" width="30%"></img>

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
|-- 16060224_haggling1
            |   |-- hdImgs
            |   |-- hdvideos
            |   |-- hdPose3d_stage1_coco19
            |   |-- calibration_160224_haggling1.json
|-- 160226_haggling1  
            |-- ...
```

Generated Annotations
=============

#### - Generating annotation json files and images with joints 3d & joints 2d & bbox & bbox_vis visualized

#### - Format of annotation json file
 - "bodies": information of people
   - "view_id": view id (HD camera id)
   - "id": person id
   - "num_person": number of the people
   - "input_width": image width (1920)
   - "input_height": image height (1080)
   - "transformed_joints_3d": GT transformed joints 3d,
   - "transformed_joints_3d_vis": visualization flags of joints 3d,
   - "projected_joints_2d": GT joints 2d projected by joints 3d using camera parameters in each view
   - "projected_joints_2d_vis": visualization flags of joints 2d,
   - "bbox": bounding boxes created by adding and subtracting an offset from the min/maxvalues of x and y values of each person's GT 2D keypoint
   - "bbox_clip": bbox cliped by image size,
   - "bbox_vis": bounding boxes created by adding and subtracting an offset from the min/max values of x and y values of each person's GT 2D keypoint that visualization flag value is true
   - "bbox_vis_clip": bbox_vis cliped by image size

#### - Annotation structures

```
|-- 16060224_haggling1
            |   |-- 00_01
            |   |   |   |-- annotations
            |   |   |   |-- images
            |   |-- 00_02
|-- 160226_haggling1  
            |-- ...
```

#### - Refer to configuration file

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
  - Refer to config file

```
python main.py --panoptic_config_file_path ./Panoptic_configs/Panoptic_annotations_builder_config.yaml
```

#### - Download Panoptic dataset
  - Select the dataset and camera number to download by editing the values of list in line 11 & 12

```
apt-get install wget
cd ./Panoptic_download_toolbox_scripts
./getData_list.sh
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
