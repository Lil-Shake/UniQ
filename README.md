## UniQ

This is the code for our paper on ACM MM 2024 " [UniQ: Unified Decoder with Task-specific Queries for Efficient Scene Graph Generation](http://arxiv.org/abs/2501.05687 "UniQ")" 

## Installation

* Create a conda environment

  ```
  conda create -n UniQ python==3.8.5
  ```

* Activate the environment

  ```
  conda activate UniQ
  ```

* Install packages

  ```
  h5py==3.10.0
  imantics==0.1.12
  easydict==1.11
  scikit-learn==1.3.2
  scipy==1.10.1
  pandas==2.0.3
  detectron2==0.6
  torch==2.2.0+cu118
  torchvision==0.17.0+cu118
  ```

## Dataset

We adopt a subset of Visual Genome VG150 that contains the most frequent 150 object classes and 50 predicate classes.

* Download Visual Genome
  * [VG_100K part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)
  * [VG_100K part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
  * [image_data.json](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/tree/master/datasets/vg)
  * [VG-SGG-dicts-with-attri.json](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/tree/master/datasets/vg)
  * [VG-SGG-with-attri.h5](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md)

```
dataset/
└── vg/
	├── VG_100K/
	├── image_data.json
	├── VG-SGG-dicts-with-attri.json
	└── VG-SGG-with-attri.h5
```

## Training

* Train unbiased UniQ (with $\alpha = 0.07$, $\beta = 0.75$). Train unbiased UniQ with more number of query groups by setting `MODEL.DETR.GROUP_DETR` to a costumed number.

  ```
  bash train_UniQ.sh
  ```

* STS, STT, TST baselines can be trained with their corresponding config files in `configs/` .

## Evaluation

* Evaluate unbiased UniQ

  ```
  bash test_UniQ.sh
  ```

* Evaluate unbiased UniQ with top-k links by setting `MODEL.DETR.MATCHER_TOPK 3`

## Visualization
![vis_00](https://github.com/user-attachments/assets/bc6f19cf-c18c-4cbc-b6a8-1947ea8f29c7)


## Citation

```
@inproceedings{10.1145/3664647.3681542,
    title = {UniQ: Unified Decoder with Task-specific Queries for Efficient Scene Graph Generation},
    author = {Liao Xinyao and Wei Wei and Chen Dangyang and Fu Yuanyuan},
    booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
    pages = {8815–8824},
    year = {2024}
}
```

## Acknowledgment

* This repository is built upon the [Iterative Scene Graph Generation](https://github.com/ubc-vision/IterativeSG) developed by Siddhesh Khandelwal and Leonid Sigal. Thanks for their extraordinary contribution to this field.
* This repository also extended [DETR](https://github.com/facebookresearch/detr) and [Group DETR](https://github.com/Atten4Vis/ConditionalDETR/tree/GroupDETR) to SGG.
