# -*- coding: utf-8 -*-
# @Time    : 2021/7/31 下午1:52
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : cityscape2voc.py
# @Software: PyCharm

"""
    这是将Cityscapes数据集转换为VOC数据集格式的工具脚本。
    根据开发需要修改cityscape_dataset_dir，voc_dataset_dir，class_names即可。

    其中:
        - cityscape_dataset_dir是原生cityscapes数据集目录路径；
        - voc_dataset_dir是VOC数据集格式的cityscapes数据集目录路径；
        - class_names为指定目标名称数组，默认为['car', 'person', 'rider', 'truck', 'bus',
          'train', 'motorcycle', 'bicycle','traffic sign','traffic light'],即只处理class_names出现的目标实例；
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

def cityscapes2voc(cityscapes_dataset_dir,voc_dataset_dir,
                  class_names=['car','person','rider','truck','bus','train','motorcycle','bicycle','traffic sign','traffic light']):
    """
    这是Cityscapes数据集转化为VOC数据集格式的函数
    :param cityscapes_dataset_dir: cityscapes数据集目录
    :param voc_dataset_dir: VOC数据集目录
    :param class_names: 目标名称数组，默认为['car','person','rider','truck','bus','train',
                                            'motorcycle','bicycle','traffic sign','traffic light']
    :return:
    """
    # 初始化cityscapes数据集相关路径
    cityscapes_image_dir = os.path.join(cityscapes_dataset_dir,"leftImg8bit")
    cityscapes_label_dir = os.path.join(cityscapes_dataset_dir,'gtFine')

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

    # 初始化图像和标签文件数组
    cityscapes_image_paths = []
    cityscapes_json_paths = []
    voc_image_paths = []
    voc_annotation_paths = []
    for choice in ['train','val']:
        # 初始化每个子数据集相关数组
        _cityscapes_image_paths = []
        _cityscapes_json_paths = []
        _cityscapes_foggy_mask_paths = []
        _voc_image_paths = []
        _voc_annotation_paths = []
        # 初始化子数据集目录路径
        _cityscapes_image_dir = os.path.join(cityscapes_image_dir, choice)
        for city_name in tqdm(os.listdir(_cityscapes_image_dir)):              # 遍历每个城市文件夹
            # 初始化每个城市目录路径
            city_dir = os.path.join(_cityscapes_image_dir,city_name)
            for image_name in os.listdir(city_dir):                     # 遍历每个城市下的所有图片
                # 初始化cityscape数据集和voc数据集图像和标签文件路径
                name,ext = os.path.splitext(image_name)
                cityscapes_image_path = os.path.join(city_dir,image_name)
                cityscapes_json_path = os.path.join(cityscapes_label_dir,choice,city_name,
                                                    name.replace("leftImg8bit","gtFine_polygons")+".json")
                if is_conitain_object(cityscapes_json_path,class_names):            # 判断图片是否包含候选目标实例
                    _cityscapes_image_paths.append(cityscapes_image_path)
                    _cityscapes_json_paths.append(cityscapes_json_path)
                    _voc_image_paths.append(os.path.join(voc_image_dir,image_name))
                    _voc_annotation_paths.append(os.path.join(voc_annotation_dir,name+".xml"))
        # 将子数据及图像名称写入voc数据集的txt文件
        with open(os.path.join(voc_imagesets_main_dir, choice + '.txt'), "w+") as f:
            for voc_image_path in _voc_image_paths:
                dir,image_name = os.path.split(voc_image_path)
                name,ext = os.path.splitext(image_name)
                f.write(name+"\n")
        cityscapes_image_paths.append(_cityscapes_image_paths)
        cityscapes_json_paths.append(_cityscapes_json_paths)
        voc_image_paths.append(_voc_image_paths)
        voc_annotation_paths.append(_voc_annotation_paths)
    cityscapes_image_paths = np.concatenate(cityscapes_image_paths,axis=0)
    cityscapes_json_paths = np.concatenate(cityscapes_json_paths,axis=0)
    voc_image_paths = np.concatenate(voc_image_paths,axis=0)
    voc_annotation_paths = np.concatenate(voc_annotation_paths, axis=0)
    # 将数据集图像名称写入voc数据集的trainval.txt文件
    with open(os.path.join(voc_imagesets_main_dir, 'trainval.txt'), "w+") as f:
        for voc_image_path in voc_image_paths:
            dir, image_name = os.path.split(voc_image_path)
            name, ext = os.path.splitext(image_name)
            f.write(name + "\n")

    # 利用多线程将cityscape数据集转化为voc数据集
    size = len(cityscapes_image_paths)
    batch_size = size // (cpu_count() - 1)
    pool = Pool(processes=cpu_count()-1)
    for i,start in enumerate(np.arange(0,size,batch_size)):
        # 获取小批量数据
        end = int(np.min([start+batch_size,size]))
        batch_cityscapes_image_paths = cityscapes_image_paths[start:end]
        batch_cityscapes_json_paths = cityscapes_json_paths[start:end]
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        print("线程{}处理{}张图像及其标签".format(i,len(batch_cityscapes_json_paths)))
        pool.apply_async(batch_image_label_process,error_callback=print,
                         args=(batch_cityscapes_image_paths,batch_cityscapes_json_paths,
                               batch_voc_image_paths,batch_voc_annotation_paths,class_names))
    pool.close()
    pool.join()

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def is_conitain_object(cityscapes_json_path,class_names):
    """
    这是判断cityscapes数据集的json标签中是否包含候选目标实例的函数
    :param cityscapes_json_path: cityscapes数据集的json标签文件路径
    :param class_names: 目标分类数组
    :return:
    """
    json_dict = json.load(open(cityscapes_json_path, 'r'))  # 加载json标签
    flag = False
    for obj in json_dict['objects']:  # load_dict['objects'] -> 目标的几何框体
        obj_label = obj['label']  # 目标的类型
        if obj_label not in class_names:
            continue
        else:
            flag = True
            break
    return flag

def find_box(points):
    """
    这是通过多边形分割标签数组获取目标检测矩形框的函数
    :param points: 多边形分割标签数组
    :return:
    """
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    xmin = int(np.min(x))
    ymin = int(np.min(y))
    xmax = int(np.max(x))
    ymax = int(np.max(y))
    return xmin,ymin,xmax,ymax

def single_image_label_process(cityscapes_image_path,cityscapes_json_path,
                               voc_image_path,voc_annotation_path,class_names):
    """
    这是将单张cityscape图像及其JSON标签转换为VOC数据集格式的函数
    :param cityscapes_image_path: cityscape图像路径
    :param cityscapes_json_path: cityscape标签路径
    :param voc_image_path: voc图像路径
    :param voc_annotation_path: voc标签路径
    :param class_names: 目标名称数组
    :return:
    """
    # 初始化VOC标签写类
    image = cv2.imread(cityscapes_image_path)
    h, w, c = np.shape(image)
    dir,image_name = os.path.split(cityscapes_image_path)
    writer = Writer(image_name,w,h)

    # JSON标签转化为VOC标签
    json_dict = json.load(open(cityscapes_json_path,'r'))            # 加载json标签
    flag = False
    for obj in json_dict['objects']:  # load_dict['objects'] -> 目标的几何框体
        obj_label = obj['label']  # 目标的类型
        if obj_label in ['out of roi', 'ego vehicle']:  # 直接跳过这两种类型 注意测试集里只有这两种类型 跳过的话测试集合里将为空的标签
            continue
        elif obj_label not in class_names:      # 目标分类标签不在候选目标分类标签数组中，跳过，只处理候选目标名称数组里的目标实例
            continue
        else:
            # 获取目标的矩形框标签,并添加到写类中
            if obj_label in ['traffic sign','traffic light']:           # 对于有空格的标签名称，删除空格
                strs = obj_label.split(" ")
                obj_label = strs[0]+"_"+strs[1]
            xmin,ymin,xmax,ymax = find_box(obj['polygon'])
            writer.addObject(obj_label,xmin,ymin,xmax,ymax)
            writer.save(voc_annotation_path)
            flag = True
    if flag:        # 写入目标标签则复制图像，否则不复制图像
        cv2.imwrite(voc_image_path, image)


def batch_image_label_process(batch_cityscapes_image_paths,batch_cityscapes_json_paths,
                              batch_voc_image_paths,batch_voc_annotation_paths,class_names):
    """
    批量处理cityscape数据转化为voc数据
    :param batch_cityscapes_image_paths: 批量cityscapes图像路径数组
    :param batch_cityscapes_json_paths: 批量cityscapes标签路径数组
    :param batch_voc_image_paths: 批量voc图像路径数组
    :param batch_voc_annotation_paths: 批量voc标签路径数组
    :param class_names: 目标名称数组
    :return:
    """
    size = len(batch_cityscapes_image_paths)
    for i in tqdm(np.arange(size)):
        cityscapes_image_path = batch_cityscapes_image_paths[i]
        cityscapes_json_path = batch_cityscapes_json_paths[i]
        voc_image_path = batch_voc_image_paths[i]
        voc_annotation_path = batch_voc_annotation_paths[i]
        single_image_label_process(cityscapes_image_path,cityscapes_json_path,
                                   voc_image_path,voc_annotation_path,class_names)

def run_main():
    """
    这是主函数
    """
    # Cityscapes --> VOC
    print("Cityscapes --> VOC Start")
    cityscape_dataset_dir = os.path.abspath("../object_detection_dataset/cityscapes")
    voc_dataset_dir = os.path.abspath("../dataset/Cityscapes")
    class_names = ['car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle','traffic sign','traffic light']
    cityscapes2voc(cityscape_dataset_dir,voc_dataset_dir,class_names)
    print("Cityscapes --> VOC Finish")

if __name__ == '__main__':
    run_main()