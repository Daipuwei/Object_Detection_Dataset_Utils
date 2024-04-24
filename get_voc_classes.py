# -*- coding: utf-8 -*-
# @Time    : 2021/9/1 17:38
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : get_voc_classes.py
# @Software: PyCharm

"""
    这是利用多线程获取VOC数据集目标分类名称数组的函数
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

def get_voc_classes(voc_dataset_dir,class_names_txt_path):
    """
    这是获取VOC数据集的目标分类数组的函数
    Args:
        voc_dataset_dir:  VOC数据集目录
        class_names_txt_path: 目标类别txt文件路径
    Returns:
    """
    # 初始化VOC数据集XML标签文件路径
    annotation_paths = []
    annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    for annotation_name in os.listdir(annotation_dir):
        annotation_paths.append(os.path.join(annotation_dir, annotation_name))
    annotation_paths = np.array(annotation_paths)

    # 利用异步多线程获取每个小批量数据集包含的目标分类名称
    size = len(annotation_paths)
    batch_size = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    results = []
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_annotation_paths = annotation_paths[start:end]
        results.append(pool.apply_async(get_batch_annotation_classes,error_callback=print_error,
                                        args=(batch_annotation_paths,)))
    pool.close()
    pool.join()

    # 合并每个小批量数据集所包含的批量数据名称
    classes_set = set()
    for result in results:
        classes_set = classes_set.union(result.get())
    classes = list(classes_set)

    # 将目标类别写入txt
    with open(class_names_txt_path,'w' ,encoding='utf-8') as f:
        for class_name in classes:
            f.write("{}\n".format(class_name))

    return classes

def parse_xml_classes(xml_path):
    """
    这是解析XML文件中包含所有目标分类的函数
    Args:
        xml_path: XML文件路径
    Returns:
    """
    tree = ET.parse(xml_path)
    objects = []
    for obj in tree.findall('object'):
        name = obj.find('name').text
        objects.append(name)
    return objects

def get_batch_annotation_classes(batch_annotation_paths):
    """
    这是获取小批量XML标签文件中目标分类名称的函数
    Args:
        batch_annotation_paths: 小批量标注文件路径数组
    Returns:
    """
    classes_set = set()
    for i in tqdm(np.arange(len(batch_annotation_paths))):
        # 获取每个XML文件中所包含的目标分类名称
        classes_names = parse_xml_classes(batch_annotation_paths[i])
        # 更新目标分类集合
        for cls in classes_names:
            classes_set.add(cls)
    return classes_set

def print_error(value):
    """
    定义错误回调函数
    Args:
        value:
    Returns:
    """
    print("error: ", value)

def run_main():
    """
    这是主函数
    """
    dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/origin/VOC2007")
    class_names_txt_path = "./voc_names.txt"
    classes_set = get_voc_classes(dataset_dir,class_names_txt_path)
    for class_name in classes_set:
        print(class_name)

if __name__ == '__main__':
    run_main()
