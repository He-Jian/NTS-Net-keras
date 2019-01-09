import keras.backend as K
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
from nts_model import get_nts_net, ranking_loss, part_cls_loss, part_cls_acc
from generator import Generator
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# from model import ParallelModelCheckpoint
from keras.utils import multi_gpu_model
from config import *
import numpy as np
import os
import sys
import math

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
# config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def step_decay(epoch):
    drop = 0.1
    for i in range(len(lr_steps)):
        if epoch + 1 <= lr_steps[i]:
            break
    lrate = initial_learning_rate * math.pow(drop, i)
    print('epoch {0} learning rate is {1}'.format(epoch + 1, lrate))
    return lrate


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='loss', verbose=0,
                 save_best_only=True, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def main():
    assert num_gpu > 0
    multi_gpu = num_gpu > 1
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    try:
        # with tf.device('/cpu:0'):
        model = get_nts_net(batch_size=batch_size)
        print(model.summary())
        if multi_gpu:
            model_parallel = multi_gpu_model(model, gpus=num_gpu)
            model_train = model_parallel
        else:
            model_train = model
        print("** compile model with class weights **")
        # optimizer = Adam(lr=initial_learning_rate)
        optimizer = SGD(lr=initial_learning_rate, momentum=0.9, nesterov=True)
        model_train.compile(optimizer=optimizer, loss={
            "cls_pred_global": "categorical_crossentropy",
            "cls_pred_part": part_cls_loss(num_classes),
            "cls_pred_concat": "categorical_crossentropy",
            "rank_concat": ranking_loss(PROPOSAL_NUM),
        },
                            loss_weights={
                                'cls_pred_global': 1.,
                                'cls_pred_part': 1,
                                'cls_pred_concat': 1,
                                'rank_concat': 1,
                            },
                            metrics={
                                'cls_pred_global': 'accuracy',
                                'cls_pred_part': part_cls_acc,
                                'cls_pred_concat': 'accuracy',
                            }
                            )

        for layer in model_train.layers:
            print layer.name, ':', layer.losses, '\n'
        print("** create image generators **")
        train_sequence = Generator(
            root=data_root,
            is_train=True,
            batch_size=batch_size,
            target_size=image_dimension,
            num_classes=num_classes,
            proposal_num=PROPOSAL_NUM,
        )
        validation_sequence = Generator(
            root=data_root,
            is_train=False,
            batch_size=batch_size,
            target_size=image_dimension,
            num_classes=num_classes,
            proposal_num=PROPOSAL_NUM,

        )
        output_weights_path = os.path.join(output_dir, output_weights_name)
        print "** set output weights path to: ", output_weights_path, " **"
        if multi_gpu:
            checkpoint = ParallelModelCheckpoint(model, output_weights_path)
        else:
            checkpoint = ModelCheckpoint(
                str(output_weights_path),
                save_weights_only=True,
                save_best_only=False,
                verbose=1,
            )

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        lrate = LearningRateScheduler(step_decay)
        callbacks = [
            checkpoint,
            TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_reduce_lr,
            #                  verbose=1, mode="auto", min_lr=min_lr),
            lrate,
            # auroc,
        ]

        print("** start training **")
        history = model_train.fit_generator(
            generator=train_sequence,
            epochs=epochs,
            class_weight={'cls_pred_global': 'auto', # it seems auto does not work after tracing the code
                          'cls_pred_part': 'auto',
                          'cls_pred_concat': 'auto', },
            validation_data=validation_sequence,
            callbacks=callbacks,
            workers=generator_workers,
            shuffle=False,
        )

        print("** done! **")
    except Exception as e:
        print(e)
    finally:
        pass


if __name__ == "__main__":
    main()
