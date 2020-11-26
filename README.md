# YOLOv4-CSP for SVHN

This fork is customized for Street View House Number dataset. The [train](https://drive.google.com/file/d/1_-x7BHGxmtWilghAs1983hxfKQJX7rpr/view?usp=sharing), [validation](https://drive.google.com/file/d/1lyyNq4-VFci70hMzmTG_IO7V3IKHmT28/view?usp=sharing), [test](https://drive.google.com/file/d/1eQ8cv6TomX3bR_RzrGS3d_cJWC7Xc2ER/view?usp=sharing) data , and pretrained [weights](https://drive.google.com/file/d/1-TOk_hSTVv4cMHA1PIuTDUEu9PX-XfgW/view?usp=sharing) can be downloaded.



### Inference

To inference, upload the downloaded {dataset}.zip and pretrained weights to your Google drive, and execute [this](https://colab.research.google.com/drive/1gUsda5kVHcZ5SFiIyl8w-ux5TaJuvPdA?usp=sharing) Google Colab Notebook from scratch with modifying paths. For changing parameters, change data/hyp.finetune.yaml or models/yolov4-csp.cfg.



### Results

0.65395 mAP on test dataset

# YOLOv4-CSP

This is the implementation of "[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)" using PyTorch framwork.

* **2020.11.16** Now supported by [Darknet](https://github.com/AlexeyAB/darknet). `[yolo] new_coords=1` 

## Installation

```
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov4_csp -it -v your_coco_path/:/coco/ -v your_code_path/:/yolo --shm-size=64g nvcr.io/nvidia/pytorch:20.06-py3

# install mish-cuda, if you use different pytorch version, you could try https://github.com/JunnYu/mish-cuda
cd /
git clone https://github.com/thomasbrandon/mish-cuda
cd mish-cuda
python setup.py build install

# go to code folder
cd /yolo
```

## Testing

[`yolov4-csp.weights`](https://drive.google.com/file/d/1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL/view?usp=sharing)

```
# download yolov4-csp.weights and put it in /yolo/weights/ folder.
python test.py --img 640 --conf 0.001 --batch 8 --device 0 --data coco.yaml --cfg models/yolov4-csp.cfg --weights weights/yolov4-csp.weights
```

You will get the results:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.47827
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.66448
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.51928
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.30647
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.53106
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.61056
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.36823
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.60434
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.65795
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.48486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.70892
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.79914
```

## Training

```
# you can change batch size to fit your GPU RAM.
python train.py --device 0 --batch-size 16 --data coco.yaml --cfg yolov4-csp.cfg --weights '' --name yolov4-csp
```

For resume training:
```
# assume the checkpoint is stored in runs/exp0_yolov4-csp/weights/.
python train.py --device 0 --batch-size 16 --data coco.yaml --cfg yolov4-csp.cfg --weights 'runs/exp0_yolov4-csp/weights/last.pt' --name yolov4-csp --resume
```

If you want to use multiple GPUs for training
```
python -m torch.distributed.launch --nproc_per_node 4 train.py --device 0,1,2,3 --batch-size 64 --data coco.yaml --cfg yolov4-csp.cfg --weights '' --name yolov4-csp --sync-bn
```

## Citation

```
@article{wang2020scaled,
  title={{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2011.08036},
  year={2020}
}
```
