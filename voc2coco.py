# -*- coding: utf-8 -*-
# @Time    : 2024/4/14 下午4:42
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : voc2coco.py
# @Software: PyCharm

"""
    这是将VOC数据集转换为COCO数据集的脚本
"""

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from multiprocessing import Pool,cpu_count

def voc2coco(voc_dataset_dir,coco_dataset_dir,class_names,save_image=False,choices=['train','val']):
    """
    这是将VOC转换为COCO数据集的函数
    Args:
        voc_dataset_dir: VOC数据集地址
        coco_dataset_dir: COCO数据集地址
        class_names: 类别名称列表
        save_image: 是否保存图片，默认为False
        choices: 子集列表或者字符串，默认为['train','val']
    Returns:
    """
    if type(choices) == 'str':
        choices = [choices]
    # 初始化VOC数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
    voc_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")
    voc_train_txt_path = os.path.join(voc_main_dir,'train.txt')
    voc_val_txt_path = os.path.join(voc_main_dir, 'val.txt')

    # 初始化COCO数据集路径
    coco_train_image_dir = os.path.join(coco_dataset_dir,"train")
    coco_val_image_dir = os.path.join(coco_dataset_dir,'val')
    coco_annotations_dir = os.path.join(coco_dataset_dir,'annotations')
    coco_train_json_path = os.path.join(coco_annotations_dir,'train.json')
    coco_val_json_path = os.path.join(coco_annotations_dir,'val.json')
    if not os.path.exists(coco_train_image_dir):
        os.makedirs(coco_train_image_dir)
    if not os.path.exists(coco_val_image_dir):
        os.makedirs(coco_val_image_dir)
    if not os.path.exists(coco_annotations_dir):
        os.makedirs(coco_annotations_dir)

    # 初始化图像及其标签路径
    image_cnt = 0
    annotation_cnt = 0
    voc_image_paths = []
    coco_image_paths = []
    if 'train' in choices:
        print("VOC-->COCO 训练集标签转换开始")
        voc_train_image_paths, coco_train_image_paths, image_cnt, annotation_cnt \
            = xml2json(voc_image_dir,voc_annotation_dir,voc_train_txt_path,
                       coco_train_image_dir,coco_train_json_path,class_names,image_cnt,annotation_cnt)
        voc_image_paths.append(voc_train_image_paths)
        coco_image_paths.append(coco_train_image_paths)
        print("VOC-->COCO 训练集标签转换结束")
    if 'val' in choices:
        print("VOC-->COCO 验证集标签转换开始")
        voc_val_image_paths, coco_val_image_paths, image_cnt, annotation_cnt \
            = xml2json(voc_image_dir, voc_annotation_dir, voc_val_txt_path,
                       coco_val_image_dir, coco_val_json_path, class_names, image_cnt, annotation_cnt)
        voc_image_paths.append(voc_val_image_paths)
        coco_image_paths.append(coco_val_image_paths)
        print("VOC-->COCO 验证集标签转换结束")
    voc_image_paths = np.concatenate(voc_image_paths)
    coco_image_paths = np.concatenate(coco_image_paths)

    # 多线程复制图像
    if save_image:
        print("开始复制图片")
        size = len(coco_image_paths)
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
            batch_coco_image_paths = coco_image_paths[start:end]
            batch_voc_image_paths = voc_image_paths[start:end]
            pool.apply_async(batch_image_copy,error_callback=print_error,
                             args=(batch_voc_image_paths,batch_coco_image_paths))
        pool.close()
        pool.join()
        print("结束复制图片")

def xml2json(voc_image_dir,voc_annotation_dir,voc_txt_path,coco_image_dir,coco_json_path,class_names,image_cnt,annotation_cnt):
    """
    这是将VOC数据集的XML标签转换为COCO数据集JSON标签的函数
    Args:
        voc_image_dir: VOC数据集图像文件夹路径
        voc_annotation_dir: VOC数据集标签文件夹路径
        voc_txt_path: VOC数据集子集txt文件路径
        coco_image_dir: COCO数据集图像文件夹路径
        coco_json_path: COCO数据集JSON文件路径
        class_names: 目标分类名称数组
        image_cnt: 图片计数器
        annotation_cnt：标签计数器
    Returns:
    """
    voc_dataset_dir,_ = os.path.split(voc_image_dir)
    _,voc_dataset_name = os.path.split(voc_dataset_dir)
    # 初始化相关文件路径
    voc_image_paths = []
    voc_image_names = []
    voc_xml_paths = []
    coco_image_paths = []
    coco_image_ids = []
    with open(voc_txt_path, 'r') as f:
        for line in f.readlines():
            voc_image_names.append("{0}.jpg".format(line.strip()))
            voc_image_paths.append(os.path.join(voc_image_dir, "{0}.jpg".format(line.strip())))
            voc_xml_paths.append(os.path.join(voc_annotation_dir, "{0}.xml".format(line.strip())))
            coco_image_paths.append(os.path.join(coco_image_dir, "{0}.jpg".format(line.strip())))
            coco_image_ids.append(image_cnt)
            image_cnt += 1
    voc_image_names = np.array(voc_image_names)
    voc_image_paths = np.array(voc_image_paths)
    voc_xml_paths = np.array(voc_xml_paths)
    coco_image_paths = np.array(coco_image_paths)
    coco_image_ids = np.array(coco_image_ids)

    image_infos = []
    detection_results = []
    for i in tqdm(np.arange(len(voc_image_paths))):
        voc_image_name = voc_image_names[i]
        voc_xml_path = voc_xml_paths[i]
        coco_image_id = coco_image_ids[i]
        if is_contain_object(voc_xml_path):
            objects,(h,w) = parse_xml(voc_xml_path)
            image_infos.append({'file_name': voc_image_name, 'id': coco_image_id,'width':w,'height':h})
            for obj in objects:
                cls_name, xmin, ymin, xmax, ymax = obj
                w = xmax - xmin
                h = ymax - ymin
                detection_results.append({'image_id': coco_image_id,
                                          'iscrowd': 0,
                                          'bbox': [xmin,ymin, w, h],
                                          'area': int(w * h),
                                          "category_id": class_names.index(cls_name),
                                          'id': annotation_cnt})
                annotation_cnt += 1
    gt_result = {}
    gt_result['images'] = image_infos
    gt_result["annotations"] = detection_results
    gt_result["categories"] = [{"id": id, "name": cls_name} for id,cls_name in enumerate(class_names)]
    gt_result_json_data = json.dumps(gt_result, indent=4, separators=(',', ': '), cls=NpEncoder)
    with open(coco_json_path, 'w+', encoding="utf-8") as f:
        f.write(gt_result_json_data)
    return voc_image_paths,coco_image_paths,image_cnt,annotation_cnt

def is_contain_object(xml_path):
    """
    这是判断XML文件中是否包含目标标签的函数
    :param xml_path: XML文件路径
    :return:
    """
    # 获取XML文件的根结点
    root = ET.parse(xml_path).getroot()
    return len(root.findall('object')) > 0

def parse_xml(xml_path):
    """
    这是解析VOC数据集XML标签文件，获取每个目标分类与定位的函数
    :param xml_path: XML标签文件路径
    :return:
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
        objects.append([obj_name, int(xmin), int(ymin), int(xmax), int(ymax)])
    return objects,(h,w)

def single_image_copy(voc_image_path,coco_image_path):
    """
    这是单张图像复制函数
    Args:
        voc_image_path: VOC图像路径
        coco_image_path: COCO图像路径
    Returns:
    """
    # 初始化VOC标签写类
    image = cv2.imread(voc_image_path)
    cv2.imwrite(coco_image_path, image)

def batch_image_copy(batch_voc_image_paths,batch_coco_image_paths):
    """
    批量复制图像的函数
    Args:
        batch_voc_image_paths: 批量voc图像路径数组
        batch_coco_image_paths: 批量coco图像路径数组
    Returns:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        coco_image_path = batch_coco_image_paths[i]
        voc_image_path = batch_voc_image_paths[i]
        single_image_copy(voc_image_path,coco_image_path)

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def get_classes(classes_path):
    """
    这是获取目标分类名称的函数
    Args:
        classes_path: 目标分类名称txt文件路径
    Returns:
    """
    classes_names = []
    with open(classes_path, 'r') as f:
        for line in f.readlines():
            classes_names.append(line.strip())
    return classes_names

class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return list(obj)
        elif isinstance(obj,bytes):
            return str(obj,encoding='utf-8')
        else:
            return super(NpEncoder, self).default(obj)

def run_main():
    """
    这是主函数
    """
    voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/coco2017_voc")
    coco_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/coco2017")
    class_name_path = os.path.abspath("./coco_names.txt")
    save_image = True
    choices = ['train','val']

    # VOC数据集转COCO数据集
    class_names = get_classes(class_name_path)
    voc2coco(voc_dataset_dir,coco_dataset_dir,class_names,save_image,choices)

if __name__ == '__main__':
    run_main()
