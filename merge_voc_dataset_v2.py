# @Time    : 2024/4/21 下午1:52
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : merge_voc_dataset_v2.py
# @Software: PyCharm

"""
    这是合并多VOC数据集的脚本
"""

import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pascal_voc_writer import Writer

def merge_voc_datasets(origin_voc_dataset_dirs, merge_voc_dataset_dir, choice_dict):
    """
    这是将多个不同VOC检测数据集合并成一个VOC检测数据集的函数
    Args:
        origin_voc_dataset_dirs: 原始VOC检测数据集地址数组
        merge_voc_dataset_dir：归并后VOC检测数据集地址
        choice_dict: 原始VOC检测数据集子集和合并VOC检测子集对应字典
    Returns:
    """
    # 初始化合并数据集相关路径
    merge_voc_image_dir = os.path.join(merge_voc_dataset_dir, 'JPEGImages')
    merge_voc_annotation_dir = os.path.join(merge_voc_dataset_dir, 'Annotations')
    merge_voc_main_dir = os.path.join(merge_voc_dataset_dir, "ImageSets", "Main")
    if not os.path.exists(merge_voc_image_dir):
        os.makedirs(merge_voc_image_dir)
    if not os.path.exists(merge_voc_annotation_dir):
        os.makedirs(merge_voc_annotation_dir)
    if not os.path.exists(merge_voc_main_dir):
        os.makedirs(merge_voc_main_dir)

    # 遍历整个子集对应字典
    print("开始遍历所有数据集，初始化所有VOC数据集图像及其标签路径")
    origin_voc_image_paths = []
    origin_voc_annotation_paths = []
    merge_voc_image_paths = []
    merge_voc_annotation_paths = []
    for merge_choice, origin_choice_list in choice_dict.items():
        merge_voc_txt_path = os.path.join(merge_voc_main_dir,merge_choice+".txt")
        with open(merge_voc_txt_path, 'w', encoding='utf-8') as f:
            # 遍历各个数据集指定子集
            for origin_voc_dataset_dir, choice_list in zip(origin_voc_dataset_dirs, origin_choice_list):
                origin_voc_image_dir = os.path.join(origin_voc_dataset_dir,"JPEGImages")
                origin_voc_annotation_dir = os.path.join(origin_voc_dataset_dir,"Annotations")
                origin_voc_main_dir = os.path.join(origin_voc_dataset_dir, "ImageSets","Main")
                if len(choice_list) > 0:
                    for origin_choice in choice_list:
                        origin_voc_txt_path = os.path.join(origin_voc_main_dir, origin_choice + ".txt")
                        with open(origin_voc_txt_path, 'r',encoding='utf-8') as g:
                            for line in tqdm(g.readlines()):
                                image_name = line.strip()
                                origin_voc_image_path = os.path.join(origin_voc_image_dir,image_name+".jpg")
                                origin_voc_annotation_path = os.path.join(origin_voc_annotation_dir, image_name + ".xml")
                                merge_voc_image_path = os.path.join(merge_voc_image_dir,image_name+".jpg")
                                merge_voc_annotation_path = os.path.join(merge_voc_annotation_dir,image_name+".xml")
                                if os.path.exists(origin_voc_image_path):
                                    origin_voc_image_paths.append(origin_voc_image_path)
                                    origin_voc_annotation_paths.append(origin_voc_annotation_path)
                                    merge_voc_image_paths.append(merge_voc_image_path)
                                    merge_voc_annotation_paths.append(merge_voc_annotation_path)
                                    f.write("{0}\n".format(image_name))
    origin_voc_image_paths = np.array(origin_voc_image_paths)
    origin_voc_annotation_paths = np.array(origin_voc_annotation_paths)
    merge_voc_image_paths = np.array(merge_voc_image_paths)
    merge_voc_annotation_paths = np.array(merge_voc_annotation_paths)
    print("结束遍历所有数据集，共计{0}张图像及其标签需要处理".format(len(origin_voc_image_paths)))

    # 多线程复制图像
    size = len(origin_voc_image_paths)
    # batch_size = size // cpu_count()
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
    for start in np.arange(0, size, batch_size):
        end = int(np.min([start + batch_size, size]))
        batch_origin_voc_image_paths = origin_voc_image_paths[start:end]
        batch_origin_voc_annotation_paths = origin_voc_annotation_paths[start:end]
        batch_merge_voc_image_paths = merge_voc_image_paths[start:end]
        batch_merge_voc_annotation_paths = merge_voc_annotation_paths[start:end]
        pool.apply_async(copy_batch_images_xmls, error_callback=print_error,
                         args=(batch_origin_voc_image_paths, batch_origin_voc_annotation_paths,
                               batch_merge_voc_image_paths,batch_merge_voc_annotation_paths))
    pool.close()
    pool.join()

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


def copy_single_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path):
    """
    这是复制单张VOC图像及其对应xml标签文件的函数
    Args:
        voc_image_path: voc图像文件路径
        voc_xml_path: voc标签文件路径
        new_voc_image_path: 新voc图像文件路径
        new_voc_xml_path: 新voc标签文件路径
    Returns:
    """
    # 复制图像
    shutil.copy(voc_image_path,new_voc_image_path)

    # 解析XML文件
    objects,(h,w) = parse_xml(voc_xml_path)

    # xml文件复制
    writer = Writer(new_voc_image_path,w,h)
    for cls_name,x1,y1,x2,y2 in objects:
        writer.addObject(cls_name,x1,y1,x2,y2)
    writer.save(new_voc_xml_path)

def copy_batch_images_xmls(batch_voc_image_paths,batch_voc_annotation_paths,
                           batch_new_voc_image_paths,batch_new_voc_annotation_paths):
    """
    这是复制批量VOC图像及其对应xml标签文件的函数
    Args:
        batch_voc_image_paths: 批量voc图像文件路径数组
        batch_voc_annotation_paths: 批量voc标签文件路径数组
        batch_new_voc_image_paths: 批量新voc图像文件路径数组
        batch_new_voc_annotation_paths: 批量新voc标签文件路径数组
    Returns:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        voc_image_path = batch_voc_image_paths[i]
        voc_xml_path = batch_voc_annotation_paths[i]
        new_voc_image_path = batch_new_voc_image_paths[i]
        new_voc_xml_path = batch_new_voc_annotation_paths[i]
        copy_single_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path)

def run_main():
    """
    这是主函数
    """
    # voc07++12+coco2017
    voc_dataset_dirs = ["/home/dpw/deeplearning/dataset/voc/coco2017",
                        "/home/dpw/deeplearning/dataset/voc/voc2007",
                        "/home/dpw/deeplearning/dataset/voc/voc2012"]
    merge_voc_dataset_dir = "/home/dpw/deeplearning/dataset/voc/coco2017+voc07++12"
    choice_dict = {
        "train": [['train'],['train'],['train']],
        "test": [['val'],['test','val'],['val']]
    }
    merge_voc_datasets(voc_dataset_dirs,merge_voc_dataset_dir,choice_dict)

if __name__ == '__main__':
    run_main()
