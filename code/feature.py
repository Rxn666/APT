import pandas as pd
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import BertTokenizer, BertModel
import os
from PIL import Image
import torch
from tqdm import *
import json
import pickle


# 使用VIT对图片进行embedding
def img_embedding(VIT_processor, VIT_model, path):
    image = Image.open(path)
    inputs = VIT_processor(images=image, return_tensors="pt")
    outputs = VIT_model(**inputs)
    feature = outputs.logits.squeeze(0)
    # feature = torch.nn.Linear(1000, length)(outputs.logits.squeeze(0))
    # print("img_embedding", feature)
    return feature


# 使用bert对文本进行embedding
def text_embedding(bert_tokenizer, bert_model, sentence):
    encoded_input = bert_tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**encoded_input)
    last_hidden_states = outputs[0]
    embedding = last_hidden_states[0].mean(dim=0)
    # sentence_embedding = torch.nn.Linear(768, length)(embedding)
    # print("sentence_embedding", sentence_embedding)
    return embedding


def Attribute_processer():
    # VIT processor
    VIT_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    VIT_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # bert
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # 读取csv文件
    df = pd.read_csv('/home/yqx/yqx_softlink/data/UCF101/data_dir_list.csv')

    # 定义mask路径
    mask_root = '/home/yqx/yqx_softlink/data/UCF101/UCF101_frames_224_16_1_mask_result'

    # 定义caption路径
    caption_root = '/home/yqx/yqx_softlink/data/UCF101/UCF101_frames_224_16_1_captions'

    # 定义label路径
    label_root = '/home/yqx/yqx_softlink/data/UCF101/UCF101_frames_224_16_1_mask_result_label'

    # # 遍历每一行
    # mask_features = dict()  # 定义一个数据字典
    # caption_features = dict()
    # label_features = dict()

    p_bar = tqdm(df.iterrows(), desc="A Processing Bar Sample: ", total=df.shape[0], ncols=100)

    for index, row in p_bar:
        # 拼接完整图片路径
        mask_path = mask_root + '/' + row['folder_path'] + '/' + row['filename']
        caption_path = caption_root + '/' + row['folder_path'] + '/output.txt'
        label_path = label_root + '/' + row['folder_path'] + '/output.txt'

        # 定义id
        id = row['folder_path'] + '/' + row['filename']

        mask_features = dict()  # 定义一个数据字典
        caption_features = dict()
        label_features = dict()
        # 处理图像特征
        features = {'mask_feature': mask_features,
                    'caption_feature': caption_features,
                    'label_feature': label_features}
        mask_features[id] = img_embedding(VIT_processor, VIT_model, mask_path)

        # 处理caption特征
        with open(caption_path) as f:
            lines = f.readlines()
            for line in lines:
                if row['filename'] in line:
                    caption = line.replace(row['filename'], '')
                    caption_features[id] = text_embedding(bert_tokenizer, bert_model, caption)

        # 处理label特征
        with open(label_path) as f:
            lines = f.readlines()
            for line in lines:
                if row['filename'] in line:
                    label = line.replace(row['filename'], '')
                    label_features[id] = text_embedding(bert_tokenizer, bert_model, label)
        file_path = '/home/yqx/yqx_softlink/data/UCF101/224_features/' + row['folder_path'] + '/' + row[
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
