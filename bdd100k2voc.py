# -*- coding: utf-8 -*-
# @Time    : 2021/7/31 下午6:12
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : bdd100k2voc.py
# @Software: PyCharm

"""
    这是将BDD100k数据集转换为VOC数据集格式的脚本，根据开发需要自行修改
    bdd100k_dataset_dir、voc_dataset_dir、class_names和dataset_type即可。

    其中:
        - bdd100k_dataset_dir代表原始BDD100k数据集目录；
        - voc_dataset_dir代表VOC数据集格式的BDD100k数据集目录；
        - class_names代表目标名称数组，该参数控制VOC数据集格式的BDD100k数据集包含的目标种类，
          默认为['car', 'person', 'rider', 'truck', 'bus','train', 'motorcycle', 'bicycle','traffic sign','traffic light']；
        - dataset_type代表数据集类型，候选值有‘all’、‘daytime’和‘night’，
          ‘all’代表转化全部数据，‘daytime’代表转化白天数据，‘night’代表转化夜晚数据；
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

def bdd100k2voc(bdd100k_dataset_dir,voc_dataset_dir,dataset_type='all',
                class_names=['car','person','rider','truck','bus','train','motorcycle','bicycle','traffic sign','traffic light']):
    """
    这是将BDD100k数据集转换为VOC数据集格式的文件
    :param bdd100k_dataset_dir: bdd100k数据集目录地址
    :param voc_dataset_dir: voc数据集目录地址
    :param dataset_type: 数据集类型，默认为‘all’，即使用全部bdd100k数据集，候选值有['all','daytime','night'],
                         若为‘daytime’则是只筛选白天场景,若为‘night’则只筛选夜晚场景
    :param class_names: 目标名称数组，默认为['car','person','rider','truck','bus','train',
                                            'motorcycle','bicycle','traffic sign','traffic light']
    :return:
    """
    # 初始化bdd100k数据集下的相关路径
    bdd100k_image_dir = os.path.join(bdd100k_dataset_dir,'images','100k')
    bdd100k_label_dir = os.path.join(bdd100k_dataset_dir,'labels','100k')

    # 初始化voc数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir,'JPEGImages')
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
    voc_imagesets_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    if not os.path.exists(voc_annotation_dir):
        os.makedirs(voc_annotation_dir)
    if not os.path.exists(voc_imagesets_main_dir):
        os.makedirs(voc_imagesets_main_dir)

    # 初始化bdd100k图像和标签文件数组
    bdd100k_image_paths = []
    bdd100k_json_paths = []
    voc_image_paths = []
    voc_annotation_paths = []
    for choice in ['train','val']:
        # 初始化每个子数据集相关数组
        _bdd100k_image_paths = []
        _bdd100k_json_paths = []
        _bdd100k_scene_mask_array = []            # 是否白天还是夜晚场景掩膜数组
        _voc_image_paths = []
        _voc_annotation_paths = []
        # 初始化子数据集目录路径
        _bdd100k_image_dir = os.path.join(bdd100k_image_dir, choice)
        _bdd100k_json_dir = os.path.join(bdd100k_label_dir, choice)
        for image_name in tqdm(os.listdir(_bdd100k_image_dir)):                     # 遍历子集下的所有图片
            # 初始化_bdd100k数据集和voc数据集图像和标签文件路径
            name,ext = os.path.splitext(image_name)
            bdd100k_image_path = os.path.join(_bdd100k_image_dir,image_name)
            bdd100k_json_path = os.path.join(_bdd100k_json_dir,name+".json")
            is_contain_object, scene_type = is_conitain_object_and_scene_type(bdd100k_json_path,class_names)
            if  is_contain_object:            # 判断图片是否包含候选目标实例
                _bdd100k_image_paths.append(bdd100k_image_path)
                _bdd100k_json_paths.append(bdd100k_json_path)
                _bdd100k_scene_mask_array.append(scene_type)
                _voc_image_paths.append(os.path.join(voc_image_dir,image_name))
                _voc_annotation_paths.append(os.path.join(voc_annotation_dir,name+".xml"))
        _bdd100k_image_paths = np.array(_bdd100k_image_paths)
        _bdd100k_json_paths = np.array(_bdd100k_json_paths)
        _bdd100k_scene_mask_array = np.array(_bdd100k_scene_mask_array,dtype=np.int32)
        _voc_image_paths = np.array(_voc_image_paths)
        _voc_annotation_paths = np.array(_voc_annotation_paths)
        # 筛选不同场景的数据
        if dataset_type == 'daytime':
            mask_index = _bdd100k_scene_mask_array == 0
        elif dataset_type == 'night':
            mask_index = _bdd100k_scene_mask_array == 1
        else:
            mask_index = np.array([True]*len(_bdd100k_scene_mask_array))
        _bdd100k_image_paths = _bdd100k_image_paths[mask_index]
        _bdd100k_json_paths = _bdd100k_json_paths[mask_index]
        _voc_image_paths = _voc_image_paths[mask_index]
        _voc_annotation_paths = _voc_annotation_paths[mask_index]

        # 将子数据及图像名称写入voc数据集的txt文件
        with open(os.path.join(voc_imagesets_main_dir, choice + '.txt'), "w") as f:
            for voc_image_path in _voc_image_paths:
                dir,image_name = os.path.split(voc_image_path)
                name,ext = os.path.splitext(image_name)
                f.write(name+"\n")
        bdd100k_image_paths.append(_bdd100k_image_paths)
        bdd100k_json_paths.append(_bdd100k_json_paths)
        voc_image_paths.append(_voc_image_paths)
        voc_annotation_paths.append(_voc_annotation_paths)
    bdd100k_image_paths = np.concatenate(bdd100k_image_paths,axis=0)
    bdd100k_json_paths = np.concatenate(bdd100k_json_paths,axis=0)
    voc_image_paths = np.concatenate(voc_image_paths,axis=0)
    voc_annotation_paths = np.concatenate(voc_annotation_paths, axis=0)
    # 将数据集图像名称写入voc数据集的trainval.txt文件
    with open(os.path.join(voc_imagesets_main_dir, 'trainval.txt'), "w") as f:
        for voc_image_path in voc_image_paths:
            dir, image_name = os.path.split(voc_image_path)
            name, ext = os.path.splitext(image_name)
            f.write(name + "\n")

    # 利用多线程将cityscape数据集转化为voc数据集
    size = len(bdd100k_image_paths)
    batch_size = size // (cpu_count() - 1)
    pool = Pool(processes=cpu_count() - 1)
    for i, start in enumerate(np.arange(0, size, batch_size)):
        # 获取小批量数据
        end = int(np.min([start + batch_size, size]))
        batch_bdd100k_image_paths = bdd100k_image_paths[start:end]
        batch_bdd100k_json_paths = bdd100k_json_paths[start:end]
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        print("线程{}处理{}张图像及其标签".format(i, len(batch_bdd100k_json_paths)))
        pool.apply_async(batch_image_label_process, error_callback=print_error,
                         args=(batch_bdd100k_image_paths, batch_bdd100k_json_paths,
                               batch_voc_image_paths, batch_voc_annotation_paths, class_names))
    pool.close()
    pool.join()

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def is_conitain_object_and_scene_type(bdd100k_json_path,class_names):
    """
    这是判断bdd100k数据集的json标签中是否包含候选目标实例并判断图片是否属于白天场景的函数
    :param bdd100k_json_path: cityscapes数据集的json标签文件路径
    :param class_names: 目标分类数组
    :return:
    """
    json_dict = json.load(open(bdd100k_json_path, 'r'))  # 加载json标签
    is_contain_object = False
    for obj in json_dict['frames'][0]["objects"]:               # load_dict['objects'] -> 目标的几何框体
        obj_label = obj['category']  # 目标的类型
        if obj_label not in class_names:
            continue
        else:
            is_contain_object = True
            break
    if json_dict['attributes']["timeofday"] == "daytime":           # 白天
        scene_type = 0
    elif json_dict['attributes']["timeofday"] == "night":           # 夜晚
        scene_type = 1
    else:                                                           # 黄昏或者黎明
        scene_type = 2
    return is_contain_object,scene_type

def single_image_label_process(bdd100k_image_path,bdd100k_json_path,
                               voc_image_path,voc_annotation_path,class_names):
    """
    这是将单张bdd100k图像及其JSON标签转换为VOC数据集格式的函数
    :param bdd100k_image_path: bdd100k图像路径
    :param bdd100k_json_path: bdd100k标签路径
    :param voc_image_path: voc图像路径
    :param voc_annotation_path: voc标签路径
    :param class_names: 目标名称数组
    :return:
    """
    # 初始化VOC标签写类
    image = cv2.imread(bdd100k_image_path)
    h, w, c = np.shape(image)
    dir,image_name = os.path.split(bdd100k_image_path)
    writer = Writer(image_name,w,h)

    # JSON标签转化为VOC标签
    flag = False
    json_dict = json.load(open(bdd100k_json_path,'r'))            # 加载json标签
    for obj in json_dict["frames"][0]['objects']:  # load_dict['objects'] -> 目标的几何框体
        obj_label = obj['category']  # 目标的类型
        if obj_label not in class_names:      # 目标分类标签不在候选目标分类标签数组中，跳过，只处理候选目标名称数组里的目标实例
            continue
        else:
            # 获取目标的矩形框标签,并添加到写类中
            if obj_label in ['traffic sign','traffic light']:           # 对于有空格的标签名称，删除空格
                strs = obj_label.split(" ")
                obj_label = strs[0]+"_"+strs[1]
            # 处理bbox坐标
            xmin = max(int(obj['box2d']['x1'])-1,0)
            ymin = max(int(obj['box2d']['y1'])-1,0)
            xmax = min(int(obj['box2d']['x2'])+1,w)
            ymax = min(int(obj['box2d']['y2'])+1,h)
            writer.addObject(obj_label,xmin,ymin,xmax,ymax)
            writer.save(voc_annotation_path)
            flag = True
    if flag:        # 复制图像
        cv2.imwrite(voc_image_path, image)


def batch_image_label_process(batch_bdd100k_image_paths,batch_bdd100k_json_paths,
                              batch_voc_image_paths,batch_voc_annotation_paths,class_names):
    """
    批量处理cityscape数据，转化为voc数据
    :param batch_bdd100k_image_paths: 批量cityscapes图像路径数组
    :param batch_bdd100k_json_paths: 批量cityscapes标签路径数组
    :param batch_voc_image_paths: 批量voc图像路径数组
    :param batch_voc_annotation_paths: 批量voc标签路径数组
    :param class_names: 目标名称数组
    :return:
    """
    size = len(batch_bdd100k_image_paths)
    for i in tqdm(np.arange(size)):
        bdd100k_image_path = batch_bdd100k_image_paths[i]
        bdd100k_json_path = batch_bdd100k_json_paths[i]
        voc_image_path = batch_voc_image_paths[i]
        voc_annotation_path = batch_voc_annotation_paths[i]
        #print(bdd100k_image_path,bdd100k_json_path,voc_image_path,voc_annotation_path)
        single_image_label_process(bdd100k_image_path,bdd100k_json_path,
                                   voc_image_path,voc_annotation_path,class_names)

def run_main():
    """
    这是主函数
    """
    # BDD100k --> VOC
    print("BDD100k --> VOC Start")
    bdd100k_dataset_dir = os.path.abspath("../object_detection_dataset/BDD100K/bdd100k")
    voc_dataset_dir = os.path.abspath("../dataset/BDD100k")
    class_names = ['car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic sign',
                   'traffic light']
    dataset_type = 'all'
    bdd100k2voc(bdd100k_dataset_dir, voc_dataset_dir, dataset_type,class_names)
    print("BDD100k --> VOC Finish")

    # BDD100k --> VOC
    print("BDD100k Daytime --> VOC Start")
    bdd100k_dataset_dir = os.path.abspath("../object_detection_dataset/BDD100K/bdd100k")
    voc_dataset_dir = os.path.abspath("../dataset/BDD100k-Daytime")
    class_names = ['car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic sign',
                   'traffic light']
    dataset_type = 'daytime'
    bdd100k2voc(bdd100k_dataset_dir, voc_dataset_dir, dataset_type,class_names)
    print("BDD100k Daytime--> VOC Finish")

    # BDD100k Night --> VOC
    print("BDD100k Night --> VOC Start")
    bdd100k_dataset_dir = os.path.abspath("../object_detection_dataset/BDD100K/bdd100k")
    voc_dataset_dir = os.path.abspath("../dataset/BDD100k-Night")
    class_names = ['car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic sign',
                   'traffic light']
    dataset_type = 'night'
    bdd100k2voc(bdd100k_dataset_dir, voc_dataset_dir, dataset_type,class_names)
    print("BDD100k Night --> VOC Finish")

if __name__ == '__main__':
    run_main()
