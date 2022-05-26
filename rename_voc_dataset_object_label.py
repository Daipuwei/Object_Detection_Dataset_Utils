# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 9:17
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : rename_voc_dataset_object_label.py
# @Software: PyCharm

"""
    这是利用多线程改变VOC数据集中目标分类标签的脚本
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from pascal_voc_writer import Writer

from multiprocessing import Pool
from multiprocessing import cpu_count

def get_voc_objects(voc_annotation_path,object_names):
    """
    这是获取VOC数据集XML文件中所有目标文件的函数
    Args:
        voc_annotation_path: VOC数据集XML文件路径
        object_names: 源目标名称数组
    Returns:
    """
    # 解析XML文件
    objects = []
    with open(voc_annotation_path,'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        # 获取所有目标
        for obj in root.findall("object"):
            name = obj.find('name').text
            xmin = obj.find("bndbox").find("xmin").text
            ymin = obj.find("bndbox").find("ymin").text
            xmax = obj.find("bndbox").find("xmax").text
            ymax = obj.find("bndbox").find("ymax").text
            if name in object_names:
                objects.append([name,xmin,ymin,xmax,ymax])
    objects = np.array(objects)
    return objects

def rename_voc_dataset_object_label(voc_dataset_dir,src_object_names,target_object_name):
    """
    这是改变VOC数据集中目标分类的函数
    Args:
        voc_dataset_dir: VOC数据集文件夹路径
        src_object_names: 原始目标分类列表
        target_object_name: 改变后目标分类名称列表
    Returns:
    """
    # 初始化XML标签文件数组
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
    voc_annotation_paths = []
    for annotation_name in os.listdir(voc_annotation_dir):
        voc_annotation_paths.append(os.path.join(voc_annotation_dir,annotation_name))
    voc_annotation_paths = np.array(voc_annotation_paths)

    # 多线程更改目标名称
    size = len(voc_annotation_paths)
    batch_size = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        pool.apply_async(batch_rename_voc_object,callback=print_error,
                         args=(batch_voc_annotation_paths,src_object_names,target_object_name))
    pool.close()
    pool.join()
    # batch_rename_voc_object(voc_annotation_paths,src_object_names,target_object_name)

def rename_voc_object(voc_annotation_path,src_object_names,target_object_name):
    """
    这是重命名目标标签的函数
    Args:
        voc_annotation_path: VOC数据集XML文件路径
        src_object_names: 原始目标名称数组
        target_object_name: 重命名后目标名称
    Returns:
    """
    # 获取需要重命名的目标
    objects = get_voc_objects(voc_annotation_path,src_object_names)
    if len(objects) > 0:
        # 删除目标节点
        with open(voc_annotation_path, 'r') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in src_object_names:
                    root.remove(obj)
            tree.write(voc_annotation_path)
            h = int(root.find("size").find('height').text)
            w = int(root.find("size").find('width').text)
            writer = Writer(voc_annotation_path,w,h)
            for _, xmin, ymin, xmax, ymax in objects:
                writer.addObject(target_object_name,xmin,ymin,xmax,ymax)
            writer.save(voc_annotation_path)

def batch_rename_voc_object(batch_voc_annotation_paths,src_object_names,target_object_name):
    """
    这是批量重命名目标标签的函数
    Args:
        batch_voc_annotation_paths: 批量VOC数据集XML文件路径数组
        src_object_names: 原始目标名称数组
        target_object_name: 重命名后目标名称
    Returns:
    """
    # 遍历所有XML标签文件，重命名目标标签
    size = len(batch_voc_annotation_paths)
    for i in tqdm(np.arange(size)):
        rename_voc_object(batch_voc_annotation_paths[i],src_object_names,target_object_name)

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def run_main():
    """
    这是主函数
    """
    # BDD100k
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/BDD100k")
    src_object_names = ['rider']
    target_object_name = "non-car"
    rename_voc_dataset_object_label(voc_dataset_dir,src_object_names,target_object_name)

    # Cityscapes
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/Cityscapes")
    src_object_names = ['rider']
    target_object_name = "non-car"
    rename_voc_dataset_object_label(voc_dataset_dir,src_object_names,target_object_name)

    # Foggy_Cityscapes_beta=0.005
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/Foggy_Cityscapes_beta_0.01")
    src_object_names = ['rider']
    target_object_name = "non-car"
    rename_voc_dataset_object_label(voc_dataset_dir,src_object_names,target_object_name)

    # Foggy_Cityscapes_beta=0.01
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/Foggy_Cityscapes_beta_0.02")
    src_object_names = ['rider']
    target_object_name = "non-car"
    rename_voc_dataset_object_label(voc_dataset_dir,src_object_names,target_object_name)

    # Foggy_Cityscapes_beta=0.02
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/Foggy_Cityscapes_beta_0.005")
    src_object_names = ['rider']
    target_object_name = "non-car"
    rename_voc_dataset_object_label(voc_dataset_dir,src_object_names,target_object_name)

    # KITTI
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/KITTI")
    src_object_names = ['rider']
    target_object_name = "non-car"
    rename_voc_dataset_object_label(voc_dataset_dir,src_object_names,target_object_name)

    # # VOC2007
    # voc_dataset_dir = os.path.abspath("/data/daipuwei/ObjectDetection/dataset/Person-NonCar/VOC2007")
    # src_object_names = ['Pedestrian']
    # target_object_name = "person"
    # rename_voc_dataset_object_label(voc_dataset_dir,src_object_names,target_object_name)

if __name__ == '__main__':
    run_main()
