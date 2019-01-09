import numpy as np
import os
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize
import scipy
from random import shuffle
from keras.utils.np_utils import to_categorical
from config import *


class Generator(Sequence):
    """
    Thread-safe image generator with imgaug support
    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, root, is_train=True, batch_size=16,
                 target_size=448, num_classes=200, proposal_num=6):
        """
        """
        self.root = root
        self.is_train = is_train
        self.batch_size = batch_size
        self.target_size = target_size
        self.proposal_num = proposal_num
        self.num_classes = num_classes
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        print(len(train_file_list))
        print(len(test_file_list))

        if self.is_train:
            self.img_list = np.array(train_file_list)
            # self.img = np.array([resize(scipy.misc.imread(os.path.join(self.root, 'images', train_file)),(600,600)) for train_file in train_file_list])
            self.label = np.array([x for i, x in zip(train_test_list, label_list) if i])
        if not self.is_train:
            self.img_list = np.array(test_file_list)
            # self.img = np.array([resize(scipy.misc.imread(os.path.join(self.root, 'images', test_file)),(600,600)) for test_file in test_file_list])
            self.label = np.array([x for i, x in zip(train_test_list, label_list) if not i])
        self.steps = np.ceil(len(self.img_list) / self.batch_size)
        self.shuffle_dataset()

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.img_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.label[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_expand = np.expand_dims(batch_y, axis=-1)
        batch_y_tile = np.tile(batch_y_expand, self.proposal_num * (self.num_classes + 1))
        batch_y_part_cls = np.repeat(batch_y_expand, self.proposal_num, axis=0)
        batch_y_part_cls = to_categorical(batch_y_part_cls, num_classes=self.num_classes)
        batch_y_part_cls = np.reshape(batch_y_part_cls, (self.batch_size, -1))
        batch_y_one_hot = to_categorical(batch_y, num_classes=self.num_classes)
        # print(batch_x.shape,batch_y_one_hot.shape,batch_y_part_cls.shape,batch_y_tile.shape)
        return {'img_input': batch_x}, {"cls_pred_global": batch_y_one_hot,  # (batch_size,num_classes)
                                        "cls_pred_concat": batch_y_one_hot,  # (batch_size,num_classes)
                                        "cls_pred_part": batch_y_part_cls,  # (batch_size,proposal_num*num_classes)
                                        "rank_concat": batch_y_tile  # (batch_size, proposal_num*num_classes+1)
                                        }

    def load_image(self, image_file):
        image_path = os.path.join(self.root, 'images', image_file)
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, (crop_from, crop_from))
        return image_array

    def transform_batch_images(self, batch_x):
        assert batch_x.shape[1:] == (crop_from, crop_from, 3)
        crop_list = []
        for i in range(len(batch_x)):
            offset_x = np.random.randint(crop_from - self.target_size) if self.is_train else (
                                                                                             crop_from - self.target_size) / 2
            offset_y = np.random.randint(crop_from - self.target_size) if self.is_train else (
                                                                                             crop_from - self.target_size) / 2
            crop_list.append(batch_x[i, offset_y:offset_y + image_dimension, offset_x:offset_x + image_dimension, :])
        batch_x = np.array(crop_list)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator

        """
        return self.label[:self.steps * self.batch_size, :]

    def shuffle_dataset(self):
        '''
        shuffle data for net epoch

        '''
        index = np.arange(len(self.img_list))
        shuffle(index)
        self.img_list = self.img_list[index]
        self.label = self.label[index]

    def on_epoch_end(self):
        if self.is_train:
            self.shuffle_dataset()
