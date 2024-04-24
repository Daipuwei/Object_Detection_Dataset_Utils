# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 12:50
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : delete_voc_dataset_object.py
# @Software: PyCharm

"""
    这是删除VOC数据集中制定目标名称标签的脚本
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

def delete_voc_dataset_object(voc_dataset_dir,class_names,is_save_class_names=True):
    """
    这是删除VOC数据集中指定目标分类的函数
    Args:
        voc_dataset_dir: VOC数据集文件夹路径
        class_names: 目标名称列表
        is_class_names: 是否为需要保留下来目标名称列表标志位，默认为True
    Returns:
    """
    # 初始化XML标签文件数组
    voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    if not os.path.exists(voc_image_dir):
        voc_image_dir = os.path.join(voc_dataset_dir, 'images')
    voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    voc_image_paths = []
    voc_annotation_paths = []
    for image_name in os.listdir(voc_image_dir):
        fname,ext = os.path.split(image_name)
        voc_image_paths.append(os.path.join(voc_image_dir,image_name))
        voc_annotation_paths.append(os.path.join(voc_annotation_dir,fname+".xml"))
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)

    # 多线程删除目标名称
    size = len(voc_annotation_paths)
    if size // cpu_count() != 0:
        num_threads = cpu_count()
    elif size // (cpu_count() // 2) != 0:
        num_threads = cpu_count() // 2
    elif size // (cpu_count() // 4) != 0:
        num_threads = cpu_count() // 4
    else:
        num_threads = 1
    batch_size = size // num_threads
    pool = Pool(processes=num_threads)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        pool.apply_async(batch_delete_voc_object,callback=print_error,
                         args=(batch_voc_image_paths,batch_voc_annotation_paths,class_names,is_save_class_names))
    pool.close()
    pool.join()

def parse_xml(xml_path,class_names,is_save_class_names=True):
    """
    这是解析VOC数据集XML标签文件，获取需要保留目标的检测框数组
    Args:
        xml_path: voc数据集XML文件路径
        class_names: 目标名称列表
        is_class_names: 是否为需要保留下来目标名称列表标志位，默认为True
    Returns:
    """
    # 获取XML文件的根结点
    root = ET.parse(xml_path).getroot()
    h = int(root.find("size").find("height").text)
    w = int(root.find("size").find("width").text)
    # 遍历所有目标
    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        if is_save_class_names:
            if obj_name in class_names:
                objects.append([obj_name, int(xmin), int(ymin), int(xmax), int(ymax)])
        else:
            if obj_name not in class_names:
                objects.append([obj_name, int(xmin), int(ymin), int(xmax), int(ymax)])
    return objects,(h,w)

def delete_voc_object(voc_image_path,voc_annotation_path,class_names,is_save_class_names=True):
    """
    这是删除单张VOC图像中指定名称目标标签的函数
    Args:
        voc_image_path: VOC数据集图像文件路径
        voc_annotation_path: VOC数据集XML文件路径
        class_names: 目标名称列表
        is_class_names: 是否为需要保留下来目标名称列表标志位，默认为True
    Returns:
    """
    # 解析需要保留目标
    save_objects,(h,w) = parse_xml(voc_annotation_path,class_names,is_save_class_names)
    # 重新写入xml文件
    writer = Writer(voc_image_path,w,h)
    for cls_name,x1,y1,x2,y2 in save_objects:
        writer.addObject(cls_name,x1,y1,x2,y2)
    writer.save(voc_annotation_path)

def batch_delete_voc_object(batch_voc_image_paths,batch_voc_annotation_paths,class_names,is_save_class_names=True):
    """
    这是删除批量VOC数据集图像中指定名称目标标签的函数
    Args:
        batch_voc_image_paths: 批量VOC图像文件路径数组
        batch_voc_annotation_paths: 批量VOC数据集XML文件路径数组
        class_names: 目标名称列表
        is_class_names: 是否为需要保留下来目标名称列表标志位，默认为True
    Returns:
    """
    # 遍历所有XML标签文件，重命名目标标签
    size = len(batch_voc_annotation_paths)
    for i in tqdm(np.arange(size)):
        delete_voc_object(batch_voc_image_paths[i],batch_voc_annotation_paths[i],class_names,is_save_class_names)

def print_error(value):
    """
    定义错误回调函数
    Args:
        value: 出错误值
    Returns:
    """
    print("error: ", value)

def run_main():
    """
    这是主函数
    """
    # BDD100k
    voc_dataset_dir = os.path.abspath("../dataset/BDD100k")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # Cityscapes
    voc_dataset_dir = os.path.abspath("../dataset/Cityscapes")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # Foggy_Cityscapes_beta=0.005
    voc_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.01")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # Foggy_Cityscapes_beta=0.01
    voc_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.02")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # Foggy_Cityscapes_beta=0.02
    voc_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.005")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # KITTI
    voc_dataset_dir = os.path.abspath("../dataset/KITTI")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

if __name__ == '__main__':
    run_main()
