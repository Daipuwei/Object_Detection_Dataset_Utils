# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 下午12:52
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : coco2voc.py
# @Software: PyCharm

"""
    这是将COCO数据集转换为VOC数据集格式的脚本，根据开发需要自行修改coco_dataset_dir、voc_dataset_dir和year即可。

    其中:
        - coco_dataset_dir代表原始COCO数据集目录；
        - voc_dataset_dir代表VOC数据集格式的COCO数据集目录；
        - year代表原始COCO数据集的年份；
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pycocotools.coco import COCO
from pascal_voc_writer import Writer

def coco2voc(coco_dataset_dir,voc_dataset_dir,year=2017):
    """
    这是将COCO数据集转换为VOC数据集格式的函数
    :param coco_dataset_dir: coco数据集目录路径
    :param voc_dataset_dir: voc数据集目录路径
    :param year: 年份，默认为2017
    :return:
    """
    # 初始化coco数据集相关路径
    coco_train_image_dir = os.path.join(coco_dataset_dir,'train{0}'.format(year))
    coco_val_image_dir = os.path.join(coco_dataset_dir,"val{0}".format(year))

    # 初始化voc数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    voc_imagesets_main_dir = os.path.join(voc_dataset_dir, "ImageSets", "Main")
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    if not os.path.exists(voc_annotation_dir):
        os.makedirs(voc_annotation_dir)
    if not os.path.exists(voc_imagesets_main_dir):
        os.makedirs(voc_imagesets_main_dir)

    # 将COCO数据集JSON标签转化为VOC数据集的XML标签
    print("COCO JSON --> VOC XML Start")
    for choice in ['train','val']:
        # 初始化COCO数据集及其目标类别字典
        coco_json_label_path = os.path.join(coco_dataset_dir,"annotations","instances_{0}{1}.json".format(choice,year))
        coco = COCO(coco_json_label_path)
        cats = coco.loadCats(coco.getCatIds())  # 数据集目标类别列表
        cat_idx = {}
        for c in cats:
            cat_idx[c['id']] = c['name']
        # 遍历所有图像
        for img in tqdm(coco.imgs):
            catIds = coco.getCatIds()
            annIds = coco.getAnnIds(imgIds=[img], catIds=catIds)
            # 划分图像文件名称
            img_fname = coco.imgs[img]['file_name']
            image_fname_ls = img_fname.split('.')
            image_fname_ls[-1] = 'xml'
            if len(annIds) > 0:
                # 生成标签文件名称
                label_fname = '.'.join(image_fname_ls)
                # 将JSON目标检测标签转化为VOC数据集标签
                writer = Writer(img_fname, coco.imgs[img]['width'], coco.imgs[img]['height'])
                anns = coco.loadAnns(annIds)
                for a in anns:
                    bbox = a['bbox']
                    bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                    xmin = max(int(bbox[0])-1,0)
                    ymin = max(int(bbox[1])-1,0)
                    xmax = min(int(bbox[2])+1,coco.imgs[img]['width'])
                    ymax = min(int(bbox[3])+1,coco.imgs[img]['height'])
                    catname = cat_idx[a['category_id']]
                    writer.addObject(catname, xmin,ymin,xmax,ymax)
                    writer.save(os.path.join(voc_annotation_dir, label_fname))
    print("COCO JSON --> VOC XML Finish")

    # 获取COCO图像路径
    coco_image_paths = []
    voc_image_paths = []
    for (choice,coco_image_dir) in [("train",coco_train_image_dir),("val",coco_val_image_dir)]:
        voc_txt_path = os.path.join(voc_imagesets_main_dir,choice+".txt")
        with open(voc_txt_path,'w') as f:
            for image_name in os.listdir(coco_image_dir):
                name,ext = os.path.splitext(image_name)
                xml_path = os.path.join(voc_annotation_dir,name+".xml")
                if os.path.exists(xml_path):            # 标签文件存在则写入
                    f.write(name+"\n")
                    coco_image_paths.append(os.path.join(coco_image_dir,image_name))
                    voc_image_paths.append(os.path.join(voc_image_dir,image_name))
    with open(os.path.join(voc_imagesets_main_dir,"trainval.txt"),'w') as f:
        for coco_image_path in coco_image_paths:
            dir,image_name = os.path.split(coco_image_path)
            name,ext = os.path.splitext(image_name)
            f.write(name+"\n")

    # 多线程复制图像
    size = len(coco_image_paths)
    batch_size = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_coco_image_paths = coco_image_paths[start:end]
        batch_voc_image_paths = voc_image_paths[start:end]
        pool.apply_async(batch_image_copy,error_callback=print_error,
                         args=(batch_coco_image_paths,batch_voc_image_paths))
    pool.close()
    pool.join()

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def single_image_copy(coco_image_path,voc_image_path):
    """
    这是单张coco图像复制函数
    :param coco_image_path: coco图像路径
    :param voc_image_path: voc图像路径
    :return:
    """
    # 初始化VOC标签写类
    image = cv2.imread(coco_image_path)
    cv2.imwrite(voc_image_path, image)

def batch_image_copy(batch_coco_image_paths,batch_voc_image_paths):
    """
    批量复制coco图像的函数
    :param batch_coco_image_paths: 批量coco图像路径数组
    :param batch_voc_image_paths: 批量voc图像路径数组
    :return:
    """
    size = len(batch_coco_image_paths)
    for i in tqdm(np.arange(size)):
        cityscapes_image_path = batch_coco_image_paths[i]
        voc_image_path = batch_voc_image_paths[i]
        single_image_copy(cityscapes_image_path,voc_image_path)

def run_main():
    """
    这是主函数
    """
    # COCO --> VOC
    print("COCO2017 --> VOC Start")
    coco_dataset_dir = os.path.abspath("../origin_dataset/COCO2017")
    voc_dataset_dir = os.path.abspath("../dataset/COCO2017")
    year = 2017
    coco2voc(coco_dataset_dir,voc_dataset_dir,year)
    print("COCO2017 --> VOC Finish")

if __name__ == '__main__':
    run_main()