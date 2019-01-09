import keras.backend as K
import tensorflow as tf
# from keras.applications.resnet50 import ResNet50
from resnet50_reg import ResNet50
# from keras.applications.vgg16 import VGG16
from keras.layers import Input, Lambda, Concatenate, GlobalAveragePooling2D, Conv2D
from keras.layers.core import Dense, Reshape, Dropout
from keras.models import Model
from keras.engine.topology import Layer
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from generate_anchor import generate_default_anchor_maps
from config import *
from keras import regularizers


class CollectScores(Layer):
    def __init__(self, **kwargs):
        self.result = None
        super(CollectScores, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CollectScores, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        """
        rpn_score, selected_indices = x  # rpn_scores (batch_size,num_anchors), selected_indices (batch_size,proposal_num)
        num_anchors = tf.shape(rpn_score)[1]
        elems = tf.concat([rpn_score, tf.cast(selected_indices, tf.float32)], axis=-1)
        # topN_scores = K.stack([K.gather(rpn_score[i], selected_indices[i]) for i in range(self.batch_size)], axis=0)
        topN_scores = tf.map_fn(lambda xx: K.gather(xx[:num_anchors], tf.cast(xx[num_anchors:], tf.int32)), elems,
                                dtype=tf.float32)
        self.result = topN_scores
        return self.result

    def compute_output_shape(self, input_shape):
        return (None,) + K.int_shape(self.result)[1:]


class CollectRegions(Layer):
    def __init__(self, **kwargs):
        self.result = None
        self.anchors = generate_default_anchor_maps()
        super(CollectRegions, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CollectRegions, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        """
        selected_indices = x
        # topN_regions = K.stack([K.gather(self.anchors, selected_indices[i]) for i in range(self.batch_size)], axis=0)
        topN_regions = tf.map_fn(lambda xx: K.gather(self.anchors, tf.cast(xx, tf.int32)), selected_indices,
                                 dtype=tf.float32)
        self.result = topN_regions
        return self.result

    def compute_output_shape(self, input_shape):
        return (None,) + K.int_shape(self.result)[1:]


class NMS(Layer):
    def __init__(self, PROPOSAL_NUM=6, **kwargs):
        self.PROPOSAL_NUM = PROPOSAL_NUM
        self.result = None
        self.anchors = generate_default_anchor_maps()
        super(NMS, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NMS, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        """
        rpn_score = x
        '''selected_indices_list = [Lambda(lambda xx: tf.image.non_max_suppression(
            self.anchors,  # boxes,
            xx,  # scores,
            max_output_size=self.PROPOSAL_NUM,
            iou_threshold=0.25,
            score_threshold=float('-inf'),
            name='nms'
        ))(rpn_score[i]) for i in range(self.batch_size)]
        selected_indices = tf.stack(selected_indices_list, axis=0)'''
        selected_indices = tf.map_fn(lambda xx: tf.image.non_max_suppression(
            self.anchors,  # boxes,[[y1,x1,y2,x2],...]
            xx,  # scores,
            max_output_size=self.PROPOSAL_NUM,
            iou_threshold=0.25,
            score_threshold=float('-inf'),
            name='nms'
        ), rpn_score, dtype=tf.int32)
        self.result = selected_indices
        return self.result

    def compute_output_shape(self, input_shape):
        return (None,) + K.int_shape(self.result)[1:]


class CropImage(Layer):
    def __init__(self, pad_size=224, nbox=4,
                 img_size=224, **kwargs):
        self.nbox = nbox
        self.cw = img_size
        self.ch = img_size
        self.result = None
        self.pad_size = pad_size
        super(CropImage, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CropImage, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        Generate the crop image.
        ------------------------------------------------------------------------
        Input: the image I[bsize,w,h,3], and the crop region A[bsize,proposal_num,4],that is [(y0,x0,y1,x1...]
        Output: the image-crop image_c[bsize*#proposal_num,w,h,3]
        """
        img_input, crop_region = x[0], x[1]
        crop_region = (K.cast(crop_region, "float32") + self.pad_size)
        crop_region = K.reshape(crop_region, (-1, 4))
        crop_region_normalized = crop_region / K.cast(K.shape(img_input)[2] - 1, 'float32')
        box_ind = tf.range(tf.shape(img_input)[0])
        box_ind = K.repeat_elements(box_ind, self.nbox, axis=0)
        result = tf.image.crop_and_resize(img_input, crop_region_normalized, box_ind, [self.ch, self.cw], name="crops")
        # crop_region=crop_region*self.stride
        '''rst = []
        re = []
        for i in range(self.batch_size): # crop region from image
            for j in range(self.nbox):
                region = img_input[i,crop_region[i][j][0]:crop_region[i][j][2]+1,crop_region[i][j][1]:crop_region[i][j][3]+1,:]
                resized_region = tf.image.resize_bilinear(tf.expand_dims(region,axis=0),(self.ch,self.cw))
                rst.append(resized_region)
            nbox_region = tf.concat(rst, axis=0)
            rst = []
            re.append(nbox_region)
        result = tf.stack(re, axis=0)'''
        self.result = result
        return self.result

    def compute_output_shape(self, input_shape):
        return (None,) + K.int_shape(self.result)[1:]
        # return tuple([self.batch_size*self.nbox,self.ch,self.cw,3])


def part_cls_loss(num_classes):
    def part_cls_loss_fn(y_true, y_pred):
        # y_true:(batch_size,proposal_num*num_classes)
        # y_pred:(batch_size*proposal_num,num_classes)
        y_pred = K.reshape(y_pred, (-1, num_classes))
        y_true = K.reshape(y_true, (-1, num_classes))
        return K.mean(categorical_crossentropy(y_true, y_pred))

    return part_cls_loss_fn


def ranking_loss(PROPOSAL_NUM):  # this loss function is tricky
    def ranking_loss_fn(y_true, y_pred):
        # target_label is label for whole image,we assign this label to part_images,with shape (batch_size,num_classes)
        target_label = y_true  # (batch_size,PROPOSAL_NUM * (num_classes+1))
        target_label = K.reshape(target_label, (-1, num_classes + 1))
        target_label = K.cast(target_label[:, 0], 'int32')  # (batch_size*PROPOSAL_NUM, )
        loss = 0.0
        # y_pred contains part_pred(classification scores for topN proposals) and topN_scores(saliency score for topN proposal)
        y_pred = K.reshape(y_pred, (-1, num_classes + 1))
        part_pred = tf.slice(y_pred, [0, 0], [-1, num_classes])
        part_pred = K.reshape(part_pred, (-1, num_classes))  # (batch_size*PROPOSAL_NUM,num_classes)
        indice = tf.stack([tf.range(tf.shape(target_label)[0]), target_label], axis=-1)
        # part_pred = K.stack([part_pred[i][target_label[i]] for i in range(batch_size * PROPOSAL_NUM)],axis=0)  # (batch_size*PROPOSAL_NUM,)
        part_pred = tf.gather_nd(part_pred, indice)
        part_pred = K.reshape(part_pred, (-1, PROPOSAL_NUM))
        topN_scores = tf.slice(y_pred, [0, num_classes], [-1, -1])  # (batch_size*PROPOSAL_NUM,1)
        topN_scores = K.reshape(topN_scores, (-1, PROPOSAL_NUM))
        # actual y_pred is topN_scores, actual y_true is part_pred
        # topN_scores with shape (batch_size,PROPOSAL_NUM)
        # part_pred with shape (batch_size*PROPOSAL_NUM,num_classes),we only need the prob of gt class
        y_true = part_pred
        y_pred = topN_scores
        for i in range(PROPOSAL_NUM):
            # for example,PROPOSAL_NUM=4,y_true is [[0.3,0.5,0.1,0.2],]
            # i=0,is_greater = 1-[1,1,0,0]=[0,0,1,1]
            # y_pred=[3,8,9,1],pivot=y_pred[0]=3,y_pred - pivot + 1=[1,6,7,-2],
            # loss=relu([1,6,7,-2]*[0,0,1,1])=relu([0,0,7,-2])=[0,0,7,0],0.3 is larger than 0.1,so,y_pred[0] should greater than y_pred[2](and margin=1),
            # but it not satisfys,so there is a loss
            is_greater = 1.0 - K.cast(y_true >= K.expand_dims(y_true[:, i], axis=-1), 'float32')
            pivot = K.expand_dims(y_pred[:, i], axis=-1)
            loss_p = (y_pred - pivot + 1) * is_greater
            loss_p = K.relu(loss_p)
            loss += loss_p
        return K.mean(loss)

    return ranking_loss_fn


def part_cls_acc(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, num_classes))
    y_pred = K.reshape(y_pred, (-1, num_classes))
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def get_nts_net(batch_size=2):
    input_shape = (image_dimension, image_dimension, 3)
    img_input = Input(shape=input_shape, name='img_input')
    # input1 = Input((PROPOSAL_NUM,), name = 'ranking_loss_input')
    base_model = ResNet50(weights='imagenet',
                          include_top=False,
                          weight_decay=0,  # add weight decay
                          # input_tensor=img_input,
                          # input_shape=input_shape,
                          pooling="avg")
    layer_dict = {}
    for layer in base_model.layers:
        layer_dict[layer.name] = layer
    # g = base_model(img_input)
    global_conv_outputs = layer_dict['activation_49'].output
    # global_conv_outputs = layer_dict['block5_conv3'].output
    base_model_1 = Model(inputs=base_model.input, outputs=global_conv_outputs, name='resnet50')  # shared model
    global_conv_outputs = base_model_1(img_input)
    global_features = GlobalAveragePooling2D()(global_conv_outputs)
    global_conv_outputs_gradient_stopped = Lambda(lambda x: K.stop_gradient(x), name='stop_gradient')(
        global_conv_outputs)
    global_features = Dropout(0.5)(global_features)
    whole_image_pred = Dense(num_classes, activation="softmax",
                             kernel_regularizer=regularizers.l2(weight_decay),
                             bias_regularizer=regularizers.l2(weight_decay),
                             # activity_regularizer=regularizers.l2(weight_decay),
                             name="cls_pred_global")(global_features)

    # proposal net
    c1 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_regularizer=regularizers.l2(weight_decay),
                bias_regularizer=regularizers.l2(weight_decay),
                # activity_regularizer=regularizers.l2(weight_decay),
                name='c1')(
        global_conv_outputs_gradient_stopped)
    c2 = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',
                kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay),
                # activity_regularizer=regularizers.l2(weight_decay),
                name='c2')(c1)
    c3 = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',
                kernel_regularizer=regularizers.l2(weight_decay), bias_regularizer=regularizers.l2(weight_decay),
                # activity_regularizer=regularizers.l2(weight_decay),
                name='c3')(c2)
    p1 = Conv2D(6, (1, 1), padding='same',
                kernel_regularizer=regularizers.l2(weight_decay),
                bias_regularizer=regularizers.l2(weight_decay),
                # activity_regularizer=regularizers.l2(weight_decay),
                name='p1')(c1)
    p2 = Conv2D(6, (1, 1), padding='same',
                kernel_regularizer=regularizers.l2(weight_decay),
                bias_regularizer=regularizers.l2(weight_decay),
                # activity_regularizer=regularizers.l2(weight_decay),
                name='p2')(c2)
    p3 = Conv2D(9, (1, 1), padding='same',
                kernel_regularizer=regularizers.l2(weight_decay),
                bias_regularizer=regularizers.l2(weight_decay),
                # activity_regularizer=regularizers.l2(weight_decay),
                name='p3')(c3)
    p1_reshape = Reshape((-1,), name='p1_reshape')(p1)
    p2_reshape = Reshape((-1,), name='p2_reshape')(p2)
    p3_reshape = Reshape((-1,), name='p3_reshape')(p3)
    rpn_score = Concatenate(axis=-1, name='rpn_concat')([p1_reshape, p2_reshape, p3_reshape])
    # anchors = Lambda(lambda x: K.variable(generate_default_anchor_maps()))(K.variable(1))

    selected_indices = NMS(PROPOSAL_NUM=PROPOSAL_NUM)(rpn_score)
    topN_scores = CollectScores(name='proposal_scores')(
        [rpn_score, selected_indices])
    crop_region = CollectRegions()(selected_indices)
    # part image cls
    img_input_padded = Lambda(lambda x: K.spatial_2d_padding(x, padding=((pad_size, pad_size), (pad_size, pad_size))),
                              name='pad_image')(img_input)
    part_imgs = CropImage(img_size=PART_RESIZE, pad_size=pad_size, nbox=PROPOSAL_NUM
                          )([img_input_padded, crop_region])
    part_imgs_stop_gradient = Lambda(lambda x: K.stop_gradient(x), name='stop_gradient_part_img')(part_imgs)
    part_conv_outputs = base_model_1(part_imgs_stop_gradient)
    part_features = GlobalAveragePooling2D()(part_conv_outputs)  # (batch_size*PROPOSAL_NUM,channel)
    part_features = Dropout(0.5)(part_features)
    part_pred = Dense(num_classes, activation="softmax",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      bias_regularizer=regularizers.l2(weight_decay),
                      # activity_regularizer=regularizers.l2(weight_decay),
                      name="cls_pred_part")(part_features)
    # concat pred
    part_feature_cat = Lambda(lambda x: K.reshape(x, (-1, PROPOSAL_NUM, last_conv_channel)),
                              name='part_feature_reshape_1')(
        part_features)
    part_feature_cat = Lambda(lambda x: x[:, :CAT_NUM, :], name='part_feature_slice')(part_feature_cat)
    part_feature_cat = Reshape((-1,), name='part_feature_reshape_2')(part_feature_cat)
    concat_feature = Concatenate(axis=-1, name='feature_concat')([global_features, part_feature_cat])
    concat_feature = Dropout(0.5)(concat_feature)
    concat_pred = Dense(num_classes, activation="softmax",
                        kernel_regularizer=regularizers.l2(weight_decay),
                        bias_regularizer=regularizers.l2(weight_decay),
                        # activity_regularizer=regularizers.l2(weight_decay),
                        name="cls_pred_concat")(concat_feature)
    # in order to implement rank loss, we concat topN_scores and part_pred as one output
    part_pred_reshape = Lambda(lambda x: K.reshape(x, (-1, PROPOSAL_NUM, num_classes)), name='cls_pred_part_reshape')(
        part_pred)
    topN_scores_reshape = Lambda(lambda x: K.reshape(x, (-1, PROPOSAL_NUM, 1)), name='topN_scores_reshape')(topN_scores)
    rank_concat = Concatenate(axis=-1, name='rank_concat_0')([part_pred_reshape, topN_scores_reshape])
    rank_concat = Lambda(lambda x: K.reshape(x, (-1, PROPOSAL_NUM * (num_classes + 1))), name='rank_concat')(
        rank_concat)
    model = Model(inputs=img_input,
                  outputs=[
                      whole_image_pred,  # (batch_size,num_classes)
                      part_pred,  # (batch_size,PROPOSAL_NUM*num_classes)
                      concat_pred,  # (batch_size,num_classes)
                      rank_concat  # topN_scores#(batch_size,PROPOSAL_NUM*(num_classes+1))
                  ],
                  name='nts-net')
    # for layer in model.layers: # this method does not work
    #    layer.kernel_regularizer = regularizers.l2(weight_decay)
    #    layer.bias_regularizer = regularizers.l2(weight_decay)
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    return model


if __name__ == '__main__':
    model = get_nts_net()
