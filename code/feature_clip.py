import clip
import torch
import clip
from PIL import Image
import requests
import pandas as pd
import numpy as np
from tqdm import *
import json
import pickle
import os

def Attribute_processer():
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32",device)
    # 读取csv文件
    df = pd.read_csv('/home/yqx/yqx_softlink/data/HMDB51/output.csv')
    # 定义mask路径
    mask_root = '/home/yqx/yqx_softlink/data/HMDB51/HMDB51_frames_224_new_16_frames_mask_result'
    # 定义caption路径
    caption_root = '/home/yqx/yqx_softlink/data/HMDB51/HMDB51_frames_224_new_16_frames_captions'
    # 定义label路径
    label_root = '/home/yqx/yqx_softlink/data/HMDB51/HMDB51_frames_224_new_16_frames_mask_result_label'
    p_bar = tqdm(df.iterrows(), desc="A Processing Bar Sample: ", total=df.shape[0], ncols=100)
    for index, row in p_bar:
        # 拼接完整图片路径
        mask_path = mask_root + '/' + row['folder_path'] + '/' + row['filename']
        caption_path = caption_root + '/' + row['folder_path'] + '/output.txt'
        label_path = label_root + '/' + row['folder_path'] + '/output.txt'

        # 定义id
        iid = row['folder_path'] + '/' + row['filename']

        mask_features = dict()  # 定义一个数据字典
        caption_features = dict()
        label_features = dict()

        features = {'mask_feature': mask_features,
                    'caption_feature': caption_features,
                    'label_feature': label_features}
        # 处理图像特征
        mask_image = Image.open(mask_path)
        mask_image_process = preprocess(mask_image).unsqueeze(0).to(device)
        with torch.no_grad():
            mask_features[iid] = model.encode_image(mask_image_process).squeeze()

        # 处理caption特征
        with open(caption_path) as f:
            lines = f.readlines()
            for line in lines:
                if row['filename'] in line:
                    caption = line.replace(row['filename'], '')
                    caption_text = clip.tokenize(caption).to(device)
                    with torch.no_grad():
                        caption_features[iid] = model.encode_text(caption_text).squeeze()

        # 处理label特征
        with open(label_path) as f:
            lines = f.readlines()
            for line in lines:
                if row['filename'] in line:
                    label = line.replace(row['filename'], '')
                    label_text = 'This is a photo of' + label
                    label_text = clip.tokenize(label_text).to(device)
                    with torch.no_grad():
                        label_features[iid] = model.encode_text(label_text).squeeze()

        file_path = '/home/yqx/yqx_softlink/data/HMDB51/224_features_CLIP_new_cpu/' + row['folder_path'] + '/' + row[
            'filename'] + '.pkl'
        dir_name = os.path.dirname(file_path)
        # 判断目录是否存在,如果不存在则创建目录
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # 判断文件是否存在,如果不存在则创建文件
        if not os.path.exists(file_path):
            # 保存字典到文件
            with open(file_path, 'wb') as f:
                pickle.dump(features, f)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(features, f)


Attribute_processer()
