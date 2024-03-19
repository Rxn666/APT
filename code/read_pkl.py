# show_pkl.py
import random
import pickle

path = '/home/yqx/yqx_softlink/data/UCF101/224_features_CLIP/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/000011.jpg.pkl'

f = open(path, 'rb')
data = pickle.load(f)

# print(data['mask_feature'])

random_key = 'train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/000011.jpg'
print(random_key, data.keys())
print(data['mask_feature'][random_key].shape,
      data['caption_feature'][random_key].shape,
      data['label_feature'][random_key].shape)

