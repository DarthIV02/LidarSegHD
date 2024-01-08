# LidarSegHD
Overarching repository for LiDAR Segmentation with HD [In progress]

## Dataset:



## Baselines:

Each baseline has it's own docker image, to run simply start instance with the respective image

### [CENet](https://github.com/huixiancheng/CENet)
Can be run with the package _cenet_image:1.0_ in this repository. To start compiling run the following command:

```bash
python train_tls.py -d /root/dataset/tls/dense_dataset_semantic/ -ac config/arch/tls.yml -n res
```

### [SQN](https://github.com/QingyongHu/SQN)
Can be run with the package _cenet_image:1.3_ in this repository. To start compiling run the following command:

```bash
python main_TLS_kitti.py --mode train --gpu 0 --labeled_point 100%
```

### [Cylinder3D](https://github.com/xinge008/Cylinder3D)
Can be run with the package _cylinder3d_image:1.0_ in this repository. To start compiling run the following command:

```bash
python train_tls.py -d /root/dataset/tls/dense_dataset_semantic/ -ac config/arch/tls.yml -n res
```
