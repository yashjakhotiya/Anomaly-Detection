import pandas, logging, imageio, math
import tensorflow as tf
import numpy as np
from PIL import Image 
from hyperparams import Hyperparams
from paths import Paths
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf

np.random.seed(5)

H = Hyperparams()
P = Paths()

img_filenames = os.listdir(P.dataset)
parsed_filenames = list(map(lambda x: x.split('_'), img_filenames))
num_samples = len(img_filenames)

cnn_train_filenames, cnn_test_filenames = train_test_split(img_filenames, test_size=H.test_size, shuffle=True)
cnn_num_samples_train = len(cnn_train_filenames)
cnn_num_samples_test = len(cnn_test_filenames)

idx = 0
lstm_files = []
curr_folder = "_".join(parsed_filenames[idx][0:2])
group_of_num_frames = []
while idx < num_samples:
    if "_".join(parsed_filenames[idx][0:2]) == curr_folder and len(group_of_num_frames) < H.num_frames:
        group_of_num_frames.append(img_filenames[idx])
        idx += 1
    else:
        if len(group_of_num_frames) == H.num_frames:
            lstm_files.append(group_of_num_frames)
            curr_folder = "_".join(parsed_filenames[idx][0:2])
            group_of_num_frames = []
        else:
            curr_folder = "_".join(parsed_filenames[idx][0:2])
            group_of_num_frames = []

lstm_train_filenames, lstm_test_filenames = train_test_split(lstm_files, test_size=H.test_size, shuffle=True)
lstm_num_samples_train = len(lstm_train_filenames)
lstm_num_samples_test = len(lstm_test_filenames)

class CNN_train_data_loader(tf.keras.utils.Sequence):

    def __init__(self, batch_size):
        self.batch_size = H.train_batch_size
        self.num_samples_train = len(cnn_train_filenames)
        
    def __len__(self):
        return self.num_samples_train // self.batch_size

    def __getitem__(self, idx):
        
        img_batch = [np.array(Image.open(os.path.join(P.dataset, cnn_train_filenames[i])).resize((H.img_width, H.img_height)))
            for i in range(idx*self.batch_size, (idx+1)*self.batch_size)]
        
        img_batch = np.array(img_batch)
        img_batch = img_batch / 255
        img_batch = np.reshape(img_batch, newshape=(H.test_batch_size, H.img_height, H.img_width, 1))
        return img_batch, img_batch

class CNN_test_data_loader(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = H.train_batch_size
        self.num_samples_test = len(cnn_test_filenames)
        
    def __len__(self):
        return self.num_samples_test // self.batch_size

    def __getitem__(self, idx):
        
        img_batch = [np.array(Image.open(os.path.join(P.dataset, cnn_test_filenames[i])).resize((H.img_width, H.img_height)))
            for i in range(idx*self.batch_size, (idx+1)*self.batch_size)]
        
        img_batch = np.array(img_batch) / 255
        img_batch = np.reshape(img_batch, newshape=(H.test_batch_size, H.img_height, H.img_width, 1))
        return img_batch, img_batch

class LSTM_train_data_loader(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = H.train_batch_size
        self.num_samples_train = len(lstm_train_filenames)
        self.model = load_model(P.cnn_encoder)
        self.model._make_predict_function()
        
    def __len__(self):
        return self.num_samples_train // self.batch_size

    def __getitem__(self, idx):
        # print("idx : {}".format(idx))
        video_batch = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            frames = []
            for j in range(H.num_frames):
                frames.append(np.array(Image.open(os.path.join(P.dataset, lstm_train_filenames[i][j])).resize((H.img_width, H.img_height))) / 255)
            frames = self.model.predict_on_batch(np.reshape(np.array(frames), newshape=(H.num_frames, H.img_height, H.img_width, 1)))
            video_batch.append(np.array(frames))

        labels_batch = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            labels_batch.append(0)
            for filename in lstm_train_filenames[i]:
                if filename.split('_')[2] == '1':
                    labels_batch[-1] = 1
                    break

        video_batch = np.array(video_batch)
        # video_batch = np.reshape(video_batch, newshape=(H.train_batch_size, H.num_frames, H.img_height, H.img_width, 1))
        labels_batch = np.array(labels_batch)
        
        return video_batch, labels_batch

class LSTM_test_data_loader(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = H.test_batch_size
        self.num_samples_test = len(lstm_test_filenames)
        self.model = load_model(P.cnn_encoder)
        self.model._make_predict_function()

    def __len__(self):
        return self.num_samples_test // self.batch_size

    def __getitem__(self, idx):
        
        video_batch = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            frames = []
            for j in range(H.num_frames):
                try:
                    frames.append(np.array(Image.open(os.path.join(P.dataset, lstm_test_filenames[i][j])).resize((H.img_width, H.img_height))) / 255)
                except:
                    print("i : {}, j : {}, file : {}, idx : {}".format(i, j, lstm_test_filenames[i][j], idx))
            frames = self.model.predict_on_batch(np.reshape(np.array(frames), newshape=(H.num_frames, H.img_height, H.img_width, 1)))
            video_batch.append(np.array(frames))


        labels_batch = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            labels_batch.append(0)
            for filename in lstm_test_filenames[i]:
                if filename.split('_')[2] == '1':
                    labels_batch[-1] = 1
                    break

        video_batch = np.array(video_batch)
        # video_batch = np.reshape(video_batch, newshape=(H.train_batch_size, H.num_frames, H.img_height, H.img_width, 1))
        labels_batch = np.array(labels_batch)

        return video_batch, labels_batch