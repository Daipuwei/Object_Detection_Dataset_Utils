# Object_Detection_Dataset_Utils
这是各种目标检测数据集预处理工具脚本集合，主要功能实现各种数据集之间的格式格式转换。数据集包括：
- VOC
- COCO
- Cityscapes
- Foggy_Cityscapes
- BDD100k
- KITTI

# 环境
- cv2 
- json
- tqdm
- numpy
- pycocotools
- multiprocessing
- pascal_voc_writer

# 功能介绍
- `cityscapes2voc.py`是利用多线程将Cityscapes数据集转换为VOC数据集格式的脚本；
- `cityscapes2foggy_cityscapes.py`是根据雾深度图像数据集和Cityscapes数据集生成Foggy_Cityscapes数据集；

# 相关资料
