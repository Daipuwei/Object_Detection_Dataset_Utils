# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 下午12:52
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : coco2voc.py
# @Software: PyCharm

"""
    这是将COCO数据集转换为VOC数据集格式的脚本。

"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

def coco2voc(coco_dataset_dir,voc_dataset_dir,year=2017):
    """
    这是将COCO数据集转换为VOC数据集格式的函数
    Args:
        coco_dataset_dir: coco数据集目录路径
        voc_dataset_dir: voc数据集目录路径
        year: 年份，默认为2017
    Returns:
    """
    # 初始化coco数据集相关路径
    coco_train_image_dir = os.path.join(coco_dataset_dir,'train{0}'.format(year))
    coco_val_image_dir = os.path.join(coco_dataset_dir,"val{0}".format(year))
    coco_train_json_path = os.path.join(coco_dataset_dir, 'annotations', 'train{0}.json'.format(year))
    coco_val_json_path = os.path.join(coco_dataset_dir, 'annotations', 'val{0}.json'.format(year))
    if not os.path.exists(coco_train_image_dir):
        coco_train_image_dir = os.path.join(coco_dataset_dir, 'train')
        coco_train_json_path = os.path.join(coco_dataset_dir, 'annotations', 'train.json')
    if not os.path.exists(coco_val_image_dir):
        coco_val_image_dir = os.path.join(coco_dataset_dir, 'val')
        coco_val_json_path = os.path.join(coco_dataset_dir, 'annotations', 'val.json')
    coco_image_json_paths = [(coco_train_image_dir,coco_train_json_path),
                             (coco_val_image_dir,coco_val_json_path)]
    choices = ["train", "val"]

    # 初始化voc数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    voc_imagesets_main_dir = os.path.join(voc_dataset_dir, "ImageSets", "Main")
    _,voc_daatset_name = os.path.split(voc_dataset_dir)
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    if not os.path.exists(voc_annotation_dir):
        os.makedirs(voc_annotation_dir)
    if not os.path.exists(voc_imagesets_main_dir):
        os.makedirs(voc_imagesets_main_dir)

    # 遍历所有子集
    coco_id_category_dict = {}
    coco_image_id_dict = {}
    coco_image_gt_dict = {}
    coco_voc_image_dict = {}
    cnt = 0
    for choice,coco_image_json_path in zip(choices,coco_image_json_paths):
        print("解析{}子集".format(choice))
        coco_image_dir,coco_json_path = coco_image_json_path
        voc_txt_path = os.path.join(voc_imagesets_main_dir,choice+".txt")
        # 读取COCO JSON标签文件
        with open(coco_json_path, "r", encoding='utf-8') as f:
            with open(voc_txt_path,'w', encoding='utf-8') as g:
                json_data = json.load(f)
                image_infos = json_data['images']
                gts = json_data['annotations']
                category_id_dict_list = json_data['categories']
                # 初始化目标名称及其对应id关系
                for _dict in category_id_dict_list:
                    coco_id_category_dict[_dict['id']] = _dict["name"]
                # 初始化coco图像id与coco图像路径和voc图像及其标签路径之间的关系
                for image_info in tqdm(image_infos):
                    # 初始化coco图像领、voc图像路径和voc标签路径
                    coco_image_path = os.path.join(coco_image_dir, image_info['file_name'])
                    voc_image_path = os.path.join(voc_image_dir, "{0}_{1:07d}.jpg".format(voc_daatset_name,cnt))
                    voc_annotation_path = os.path.join(voc_annotation_dir, "{0}_{1:07d}.xml".format(voc_daatset_name, cnt))
                    coco_image_id_dict[image_info['id']] = coco_image_path
                    coco_voc_image_dict[coco_image_path] = (voc_image_path,voc_annotation_path)
                    # 将VOC图像名称写入子集txt文件中
                    g.write("{0}_{1:07d}\n".format(voc_daatset_name,cnt))
                    cnt += 1
                for gt in gts:
                    image_id = gt['image_id']
                    x1, y1, w, h = gt['bbox']
                    cls_id = gt['category_id']
                    image_path = coco_image_id_dict[image_id]
                    if image_path not in coco_image_gt_dict.keys():
                        coco_image_gt_dict[image_path] = [[coco_id_category_dict[cls_id], x1, y1, w, h]]
                    else:
                        coco_image_gt_dict[image_path].append([coco_id_category_dict[cls_id], x1, y1, w, h])

    # 初始化COCO与VOC数据集相关文件路径
    coco_image_paths = []
    coco_labels = []
    voc_image_paths = []
    voc_annotation_paths = []
    for coco_image_path,coco_label_ in coco_image_gt_dict.items():
        voc_image_path,voc_annotation_path = coco_voc_image_dict[coco_image_path]
        coco_image_paths.append(coco_image_path)
        coco_labels.append({"gt":coco_label_})
        voc_image_paths.append(voc_image_path)
        voc_annotation_paths.append(voc_annotation_path)
    coco_image_paths = np.array(coco_image_paths)
    coco_labels = np.array(coco_labels)
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)

    print("开始多线程处理COCO图像及其标签并转换VOC格式")
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
    for start in np.arange(0, size, batch_size):
        end = int(np.min([start + batch_size, size]))
        batch_coco_image_paths = coco_image_paths[start:end]
        batch_coco_labels = coco_labels[start:end]
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        pool.apply_async(batch_images_labels_process, error_callback=print_error,
                         args=(batch_coco_image_paths, batch_coco_labels,
                               batch_voc_image_paths,batch_voc_annotation_paths))
    pool.close()
    pool.join()
    print("结束多线程处理")

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def single_image_label_process(coco_image_path, coco_label,voc_image_path,voc_annotation_path):
    """
    这是对单张coco图像及其标签进行处理转换为VOC格式的函数
    Args:
        coco_image_path: coco图像路径
        coco_label: coco标签
        voc_image_path: voc图像路径
        voc_annotation_path: voc标签路径
    Returns:
    """
    # 复制图像
    image = cv2.imread(coco_image_path)
    h,w,_ = np.shape(image)
    cv2.imwrite(voc_image_path, image)

    # 将目标框标签写入xml
    writer = Writer(voc_image_path,w,h)
    for bbox_label in coco_label['gt']:
        cls_name, x1, y1, w, h = bbox_label
        x2 = x1 + w
        y2 = y1 + h
        writer.addObject(cls_name,x1,y1,x2,y2)
    writer.save(voc_annotation_path)

def batch_images_labels_process(batch_coco_image_paths, batch_coco_labels,
                                batch_voc_image_paths,batch_voc_annotation_paths):
    """
    这是对批量coco图像及其标签进行处理转换为VOC格式的函数
    Args:
        batch_coco_image_paths: 批量coco图像路径数组
        batch_coco_labels: 批量coco标签数组
        batch_voc_image_paths: 批量voc图像路径数组
        batch_voc_annotation_paths: 批量voc标签路径数组
    Returns:
    """
    size = len(batch_coco_image_paths)
    for i in tqdm(np.arange(size)):
        coco_image_path = batch_coco_image_paths[i]
        coco_label = batch_coco_labels[i]
        voc_image_path = batch_voc_image_paths[i]
        voc_annotation_path = batch_voc_annotation_paths[i]
        single_image_label_process(coco_image_path, coco_label, voc_image_path,voc_annotation_path)

def run_main():
    """
    这是主函数
    """
    # COCO --> VOC
    print("COCO2017 --> VOC Start")
    coco_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/COCO2017")
    voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/dataset/VOC/COCO2017")
    year = 2017
    coco2voc(coco_dataset_dir,voc_dataset_dir,year)
    print("COCO2017 --> VOC Finish")

if __name__ == '__main__':
    run_main()
