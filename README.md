# LidarSegHD
Overarching repository for LiDAR Segmentation with HD [In progress]

## Dataset:



## Baselines:

Each baseline has it's own docker image, to run simply start instance with the respective image

## [CENet](https://github.com/huixiancheng/CENet): 
Can be run with the package _cenet_image:1.0_ in this repository. To start compiling run the following command:

```bash
python train_tls.py -d /root/dataset/tls/dense_dataset_semantic/ -ac config/arch/tls.yml -n res
```
