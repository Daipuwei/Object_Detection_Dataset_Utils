# -*- coding: utf-8 -*-
# @Time    : 2021/8/5 下午1:52
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : copy_voc_dataset.py
# @Software: PyCharm

"""
    这是根据指定目标类别复制VOC数据集的脚本,包含对图像及其xml重命名功能
"""

import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

def copy_voc_dataset(voc_dataset_dir,new_voc_dataset_dir,class_names=None):
    """
    这是根据指定目标类别复制VOC数据集的函数
    Args:
        voc_dataset_dir: voc数据集地址
        new_voc_dataset_dir: 新voc数据集地址
        class_names: 目标类别数组,默认为None
    Returns:
    """
    # 初始化voc数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    if not os.path.exists(voc_image_dir):
        voc_image_dir = os.path.join(voc_dataset_dir, 'images')
    voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    new_voc_image_dir = os.path.join(new_voc_dataset_dir, 'JPEGImages')
    new_voc_annotation_dir = os.path.join(new_voc_dataset_dir, "Annotations")
    new_voc_imagesets_main_dir = os.path.join(new_voc_dataset_dir, "ImageSets", "Main")
    _,new_voc_dataset_name = os.path.split(new_voc_dataset_dir)
    if not os.path.exists(new_voc_image_dir):
        os.makedirs(new_voc_image_dir)
    if not os.path.exists(new_voc_annotation_dir):
        os.makedirs(new_voc_annotation_dir)
    if not os.path.exists(new_voc_imagesets_main_dir):
        os.makedirs(new_voc_imagesets_main_dir)

    # 筛选包含指定目标的VOC数据集的图像和xml文件路径
    voc_image_paths = []
    voc_annotation_paths = []
    new_voc_image_paths = []
    new_voc_annotation_paths = []
    cnt = 0
    for image_name in os.listdir(voc_image_dir):
        name,ext = os.path.splitext(image_name)
        image_path = os.path.join(voc_image_dir,image_name)
        xml_path = os.path.join(voc_annotation_dir,name+".xml")
        if is_contain_classes(xml_path,class_names):
            new_name = "{0}_{1:06d}".format(new_voc_dataset_name, cnt)
            new_image_path = os.path.join(new_voc_image_dir,new_name+".jpg")
            new_xml_path = os.path.join(new_voc_annotation_dir, new_name+".xml")
            voc_image_paths.append(image_path)
            voc_annotation_paths.append(xml_path)
            new_voc_image_paths.append(new_image_path)
            new_voc_annotation_paths.append(new_xml_path)
            cnt += 1
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    new_voc_image_paths = np.array(new_voc_image_paths)
    new_voc_annotation_paths = np.array(new_voc_annotation_paths)

    # 多线程处理复制图像和xml文件
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
        batch_new_voc_image_paths = new_voc_image_paths[start:end]
        batch_new_voc_annotation_paths = new_voc_annotation_paths[start:end]
        pool.apply_async(batch_process_image_xml,error_callback=print_error,
                         args=(batch_voc_image_paths,batch_voc_annotation_paths,
                               batch_new_voc_image_paths,batch_new_voc_annotation_paths,class_names))
    pool.close()
    pool.join()

def is_contain_classes(xml_path,class_names=None):
    """
    这是判断xml文件中是否包含指定名称标签的函数
    Args:
        xml_path: xml文件路径
        class_names: 目标分类数组，默认为None
    Returns:
    """
    if class_names is None:
        flag = True
    else:
        flag = False
    if os.path.exists(xml_path):
        # 解析XML文件
        in_file = open(xml_path)
        tree = ET.parse(in_file)
        root = tree.getroot()

        # 遍历所有目标结点,判断是否存在指定目标
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if class_names is not None:
                if cls_name in class_names:
                    flag = True
                    break
                else:
                    continue

    return flag

def print_error(value):
    """
    定义错误回调函数
    Args:
        value: 出错误值
    Returns:
    """
    print("error: ", value)

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

def process_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path,class_names):
    """
    这是复制VOC图像及其xml文件的函数
    Args:
        voc_image_path: voc图像文件路径
        voc_xml_path: voc标签文件路径
        new_voc_image_path: voc图像文件新路径
        new_voc_xml_path: voc标签文件新路径
        class_names: 目标分类数组
    Returns:
    """
    # 复制图像
    shutil.copy(voc_image_path,new_voc_image_path)

    # xml解析复制制定类别目标
    objects,(h,w) = parse_xml(voc_xml_path,class_names)
    writer = Writer(new_voc_image_path,w,h)
    for cls_name,x1,y1,x2,y2 in objects:
        writer.addObject(cls_name,x1,y1,x2,y2)
    writer.save(new_voc_xml_path)

def batch_process_image_xml(batch_voc_image_paths,batch_voc_annotation_paths,
                            batch_new_voc_image_paths,batch_new_voc_annotation_paths,class_names):
    """
    这是批量处理VOC图像及其标签函数
    Args:
        batch_voc_image_paths: 批量voc图像路径数组
        batch_voc_annotation_paths: 批量voc标签路径数组
        batch_new_voc_image_paths: 批量voc图像新路径数组
        batch_new_voc_annotation_paths: 批量voc标签新路径数组
        class_names: 目标类别数组
    Returns:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        voc_image_path = batch_voc_image_paths[i]
        voc_xml_path = batch_voc_annotation_paths[i]
        new_voc_image_path = batch_new_voc_image_paths[i]
        new_voc_xml_path = batch_new_voc_annotation_paths[i]
        process_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path,class_names)

def run_main():
    """
    这是主函数
    """
    # VOC2007 --> VOT_VOC2007
    voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/voc/voc2007")
    new_voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/voc/vot_voc2007")
    class_names = ['person','horse','cat','cow','bird','train','sleep',"bus","car","boat","aeroplane"]
    copy_voc_dataset(voc_dataset_dir, new_voc_dataset_dir, class_names,)

    # VOC2012 --> VOT_VOC2012
    voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/voc/voc2012")
    new_voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/voc/vot_voc2012")
    class_names = ['person','horse','cat','cow','bird','train','sleep',"bus","car","boat","aeroplane"]
    copy_voc_dataset(voc_dataset_dir, new_voc_dataset_dir, class_names,)

    # COCO2017 --> VOT_COCO2017
    voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/voc/coco2017")
    new_voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/voc/vot_coco2017")
    class_names = ['person',"airplane","bus","train","truck","boat","bird","cat","dog","horse","sheep","cow",
                   "elephant","bear","zebra","giraffe","fork",]
    copy_voc_dataset(voc_dataset_dir, new_voc_dataset_dir, class_names)


if __name__ == '__main__':
    run_main()
