CMU Panoptic
=============

#### - CMU Panoptic dataset

#### - CMU Panoptic dataset structure

Annotations
=============

#### - Annotation format

#### - Annotationt structure

Parameters
=============

#### - Configuration files

#### - Selecting sequences & camera views

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

```
python main.py --panoptic_config_file_path ./Panoptic_configs/Panoptic_annotations_builder_config.yaml
```

#### - Download Panoptic dataset

```
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

#### - CMU Panoptic dataset download toolbox

https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox

#### - CMU Panoptic Pytorch dataset class 

https://github.com/microsoft/voxelpose-pytorch

Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com
