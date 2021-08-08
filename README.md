# Object_Detection_Dataset_Utils
这是各种目标检测数据集预处理工具脚本集合，主要功能实现各种数据集之间的格式格式转换。数据集包括：
- VOC
- COCO
- Cityscapes
- Foggy_Cityscapes
- BDD100k
- KITTI

# 环境配置
- cv2 
- xml
- json
- tqdm
- numpy
- pycocotools
- multiprocessing
- pascal_voc_writer

# 功能介绍
- `cityscapes2voc.py`是利用多线程将Cityscapes数据集转换为VOC数据集格式的脚本；
- `cityscapes2foggy_cityscapes.py`是根据雾深度图像数据集和Cityscapes数据集生成Foggy_Cityscapes数据集；
- `bdd100k2voc.py`是利用多线程将BDD100k数据集转换VOC数据集格式的脚本；
- `kitti2voc.py`是利用多线程将KITTI数据集转换为VOC数据集格式的脚本；
- `coco2voc.py`是利用多线程将COCO数据集转换为VOC数据集格式的脚本；

# How to Use
## Cityscapes --> VOC
下载Cityscapes数据集并完成解压，其文档结构如下所示：
```
cityscapes
    - leftImg8bit
        - train
            - city1
                - image1.png
                - image2.png
                - ...
            - city2
            - ...
        - val
        - test
    - gtFine
        - train
            - city1
                - label1.json
                - label2.json
                - ...
            - city2
            - ...
        - val
        - test
```
根据开发需要修改`cityscape_dataset_dir`、`voc_dataset_dir`和`class_names`,其中`cityscape_dataset_dir`代表原始Cityscapes数据集目录，`voc_dataset_dir`代表转换为VOC数据集格式后Cityscapes数据集目录，`class_names`代表目标名称数组，然后运行如下代码即可生成VOC数据集格式的Cityscapes数据集。
```bash
python cityscape2voc.py
```

## Foggy Cityscapes --> VOC
首先下载Cityscapes的数据集和Foggy Cityscapes的图像数据集，并完成两个数据集的解压，Foggy Cityscapes图像数据集文件结构如下所示：
```
cityscapes_foggy
    - leftImg8bit_trainvaltest_foggy
        - leftImg8bit_foggy
            - train
                - city1
                    - image1_foggy_beta_0.005.png
                    - image1_foggy_beta_0.01.png
                    - image1_foggy_beta_0.02.png
                    - ...
                - city2
                - ...
            - val
            - test
```
然后根据`cityscape2voc.py`脚本将Cityscapes数据集转换为VOC数据集格式，最后根据开发需要修改`cityscapes_dataset_dir`、`foggy_cityscapes_dataset_dir`、`foggy_image_dir`和`beta`，其中`beta`控制雾浓度，候选值有`0.005`、`0.01`和`0.02`，`cityscapes_dataset_dir`为VOC数据集格式的Cityscapes数据集目录、`foggy_cityscapes_dataset_dir`为voc数据集格式的Foggy Cityscapes数据集目录、`foggy_image_dir`为Foggy Cityscapes图像数据集目录，然后运行如下代码即可生成VOC数据集格式的Foggy Cityscapes数据集。
```bash
pyhon cityscapes2foggy_cityscapes.py
```

# BDD100k --> VOC
下载BDD100k数据集并完成解压，其文件结构如下所示：
```
bdd100k
    - images
        - 10k
        - 100k
            - train
                - image1.png
                - image2.png
                - ...
            - val
            - test
    - labels
        - 10k
        - 100k
            - train
                - label1.json
                - label2.json
                - ...
            - val
            - test
```
然后根据开发需要修改`bdd100k_dataset_dir`、`voc_dataset_dir`、`class_names`和`dataset_type`,其中`bdd100k_dataset_dir`代表原始BDD100k数据集目录，`voc_dataset_dir`代表VOC数据集格式的BDD100k数据集目录，`class_names`代表目标名称数组，该参数控制VOC数据集格式的BDD100k数据集包含的目标种类，`dataset_type`代表数据集类型，候选值有‘all’、‘daytime’和‘night’，‘all’代表转化全部数据，‘daytime’代表转化白天数据，‘night’代表转化夜晚数据。最后运行如下代码即可生成VOC数据集格式的BDD100k数据集。
```bash
python bdd100k2voc.py
```

## KITTI --> VOC
首先下载KITTI数据集并完成解压，其文件结构如下所示：
```
kitti
    - train
        - image_2
            - name1.png
            - name2.png
            - ...
        - label_2
            - name1.txt
            - name2.txt
            - ...
    - test
```
然后根据开发需要自行修改`kitti_dataset_dir`、`voc_dataset_dir`、`train_ratio`和`class_names`，其中`kitti_dataset_dir`为原始KITTI数据集目录路径,`voc_dataset_dir`为VOC数据集格式的KITTI数据集目录路径,`train_ratio`为训练集比例，默认为0.8,用于随机划分训练集和验证集使用，`class_names`为目标名称数组，该参数控制VOC数据集格式的BDD100k数据集包含的目标种类，默认为['Person_sitting',"Pedestrian",'Cyclist',"Truck","Car","Tram","Van"]， 最后运行如下命令即可生成VOC数据集格式的KITTI数据集。
```bash
python kitti2voc.py
```
## COCO --> VOC
下载COCO数据集并完成解压，其文件结构如下：
```
coco
    - train
        - name1.jpg
        - name2.jpg
        - ...
    - val
    - annotations
        - instances_train_year.json
        - instances_val_year.json
        - ...
```
然后根据开发需要自行修改`coco_dataset_dir`、`voc_dataset_dir`和`year`， 其中`coco_dataset_dir`代表原始COCO数据集目录,`voc_dataset_dir`代表VOC数据集格式的COCO数据集目录,`year`代表原始COCO数据集的年份, 最后运行如下命令即可生成VOC数据集格式的COCO数据集。
```bash
python coco2voc.py
```