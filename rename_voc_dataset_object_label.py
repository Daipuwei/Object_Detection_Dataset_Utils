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

def rename_voc_dataset_object_label(voc_dataset_dir,object_name_dict):
    """
    这是改变VOC数据集中目标分类的函数
    Args:
        voc_dataset_dir: VOC数据集文件夹路径
        object_name_dict: 目标名称字典，键为更改后目标名称，值为待修改过的目标名称数组
    Returns:
    """
    # 初始化VOCs数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
    if not os.path.exists(voc_image_dir):
        voc_image_dir = os.path.join(voc_dataset_dir,"images")
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")

    # 初始化VOC图像及其标签路径
    voc_image_paths = []
    voc_annotation_paths = []
    for image_name in os.listdir(voc_image_dir):
        fname,ext = os.path.splitext(image_name)
        voc_image_paths.append(os.path.join(voc_image_dir,image_name))
        voc_annotation_paths.append(os.path.join(voc_annotation_dir,fname+".xml"))
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)

    # 多线程更改目标名称
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
        batch_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        pool.apply_async(batch_rename_voc_object,callback=print_error,
                         args=(batch_image_paths,batch_voc_annotation_paths,object_name_dict))
    pool.close()
    pool.join()
    # batch_rename_voc_object(voc_annotation_paths,src_object_names,target_object_name)

def parse_xml(xml_path,class_names=None):
    """
     这是解析VOC数据集XML标签文件，获取每个目标分类与定位的函数
    Args:
        xml_path: XML标签文件路径
        class_names: 目标名称数组，默认为None
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
        if class_names is None:
            objects.append([obj_name, int(xmin), int(ymin), int(xmax), int(ymax)])
        else:
            if obj_name in class_names:
                objects.append([obj_name, int(xmin), int(ymin), int(xmax), int(ymax)])
    return objects,(h,w)


def rename_voc_object(voc_image_path,voc_annotation_path,object_name_dict):
    """
    这是对一张VOC数据集XML标签文件重命名目标标签的函数
    Args:
        voc_image_path: VOC数据集图像路径
        voc_annotation_path: VOC数据集XML文件路径
        object_name_dict: 目标名称字典，键为更改后目标名称，值为待修改过的目标名称数组
    Returns:
    """
    # 获取XML标签
    objects,(h,w) = parse_xml(voc_annotation_path)

    # 重新写入目标标签，并完成重命名
    writer = Writer(voc_image_path,w,h)
    for cls_name,x1,y1,x2,y2 in objects:
        rename_cls_name = cls_name
        # 遍历目标名称修改字典
        for _rename_cls_name,src_cls_names in object_name_dict.items():
            # 目标名称在待修改目标名称列表中
            if cls_name in src_cls_names:
                rename_cls_name = _rename_cls_name
                break
        writer.addObject(rename_cls_name,x1,y2,x2,y2)
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
