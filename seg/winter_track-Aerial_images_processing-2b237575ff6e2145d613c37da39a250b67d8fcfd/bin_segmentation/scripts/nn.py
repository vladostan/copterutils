import keras
from keras.losses import binary_crossentropy
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, UpSampling2D, Lambda,  add, Activation, Convolution2D
from keras.layers import concatenate as koncatenate
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Reshape, SeparableConv2D, DepthwiseConv2D, Permute
from keras.models import Model
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Add

#relu6 = keras.applications.mobilenet.relu6
#_depthwise_conv_block = keras.applications.mobilenet._depthwise_conv_block

n_rows,n_cols = (64*4,192*4)

rows = 1280
cols = 1024
classes_num = 12

def get_custom_unet(input_shape = (n_rows,n_cols,1),n_conv = 8,act='relu',bottle_idx=5, bin=False):
    """
        generic unet implementation
    """

    conv = {}
    pool = {}
    up = {}

    d_rate=0.2

    ## only for simplicity
    pool[0] = Input(input_shape,name='input')
    j = bottle_idx-1
    for i in range(1,2*bottle_idx):

        if i < bottle_idx:
            # encoder
            conv[i] = Conv2D(2**(i-1)*n_conv, (3, 3), activation=act, padding='same',
                             name = 'conv_%s1' % i )(pool[i-1])
 #          conv[i] = BatchNormalization()(conv[i])
 #           conv[i] = Dropout(d_rate)(conv[i])

            conv[i] = Conv2D(2**(i-1)*n_conv, (3, 3), activation=act, padding='same',
                             name = 'conv_%s2' % i)(conv[i])
 #           conv[i] = BatchNormalization()(conv[i])
 #           conv[i] = Dropout(d_rate)(conv[i])

            pool[i] = MaxPooling2D(pool_size=(2, 2),name = 'maxpool_%s' % i)(conv[i])
        elif i == bottle_idx:
            ## bottle
            conv[bottle_idx] = Conv2D(2**(bottle_idx-1)*n_conv, (3, 3), activation=act, padding='same',
                                      name='conv_%s1' % i)(pool[bottle_idx-1])
            conv[bottle_idx] = Conv2D(2**(bottle_idx-1)*n_conv, (3, 3), activation=act,
                                      padding='same',name='conv_%s2' % i)(conv[bottle_idx])
        else:
            ## decoder
            up[i] = koncatenate([UpSampling2D(size=(2, 2),name = 'up_%s' % i)(conv[i-1]), conv[j]],
                                axis=3,name = 'concat_%s' % i)

            conv[i] = Conv2D(2**(j-1)*n_conv, (3, 3), activation=act, padding='same',
                             name = 'conv_%s1' % i)(up[i])
   #         conv[i] = BatchNormalization()(conv[i])
  #          conv[i] = Dropout(d_rate)(conv[i])

            conv[i] = Conv2D(2**(j-1)*n_conv, (3, 3), activation=act, padding='same',
                             name = 'conv_%s2' % i)(conv[i])
    #        conv[i] = BatchNormalization()(conv[i])
     #       conv[i] = Dropout(d_rate)(conv[i])

            j-=1

    global classes_num
    global rows
    global cols
#    final_conv = Conv2D(classes_num, (1, 1))(conv_finish)
#    outmap = Reshape((classes_num, rows * cols))(final_conv)
#    outmap = Permute((2, 1))(outmap)

    if bin:
       conv_finish = conv[2*bottle_idx-1]
#       conv_finish = Conv2D(256, (1, 1))(conv[2*bottle_idx-1])
       outmap = Conv2D(1, (1, 1), activation='sigmoid',name = 'outmap')(conv_finish)
    else:
       conv_finish = Conv2D(1024, (1, 1))(conv[2*bottle_idx-1])
       outmap = Conv2D(12, (1, 1), activation='softmax',name = 'outmap')(conv_finish)
#    outmap = Activation('softmax')(outmap)
 #   outmap = Reshape((cols, rows, classes_num))(outmap)
    model = Model(inputs=[pool[0]], outputs=[outmap])
    return model


def get_fpn_net(input_shape=(1024, 1280, 3), act='relu'):
    input = Input(input_shape)

    conv11 = Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), activation=act, padding='same', data_format='channels_last')(input)
    conv12 = Conv2D(filters=512, kernel_size=(2, 2), strides=(2, 2),  activation=act, padding='same', data_format='channels_last')(conv11)
    conv13 = Conv2D(filters=1024, kernel_size=(2, 2), strides=(2, 2), activation=act, padding='same', data_format='channels_last')(conv12)
    conv14 = Conv2D(filters=2048, kernel_size=(2, 2), strides=(2, 2), activation=act, padding='same', data_format='channels_last')(conv13)

    conv24 = Conv2D(256, kernel_size=(1, 1), activation=act, padding='same', data_format='channels_last')(conv14)
    conv23 = Add()([UpSampling2D(size=(2, 2), data_format='channels_last')(conv24),
                Conv2D(filters=256, kernel_size=(1, 1), activation=act, padding='same', data_format='channels_last')(conv13)])
    conv22 = Add()([UpSampling2D(size=(2, 2), data_format='channels_last')(conv23),
                Conv2D(filters=256, kernel_size=(1, 1), activation=act, padding='same', data_format='channels_last')(conv12)])
    conv21 =  Add()([UpSampling2D(size=(2, 2), data_format='channels_last')(conv22),
                Conv2D(filters=256, kernel_size= (1, 1), activation=act, padding='same', data_format='channels_last')(conv11)])

    conv34 = Conv2D(filters=128, kernel_size=(3, 3), activation=act, padding='same', data_format='channels_last')(conv24)
    conv34 = Conv2D(filters=128, kernel_size=(3, 3), activation=act, padding='same', data_format='channels_last')(conv34)

    conv33 = Conv2D(filters=128, kernel_size=(3, 3), activation=act, padding='same', data_format='channels_last')(conv23)
    conv33 = Conv2D(filters=128, kernel_size=(3, 3), activation=act, padding='same', data_format='channels_last')(conv33)

    conv32 = Conv2D(filters=128, kernel_size=(3, 3), activation=act, padding='same', data_format='channels_last')(conv22)
    conv32 = Conv2D(filters=128, kernel_size=(3, 3), activation=act, padding='same', data_format='channels_last')(conv32)

    conv31 = Conv2D(filters=128, kernel_size=(3, 3), activation=act, padding='same', data_format='channels_last')(conv21)
    conv31 = Conv2D(filters=128, kernel_size=(3, 3), activation=act, padding='same', data_format='channels_last')(conv31)

    final_conv4 = UpSampling2D((8, 8))(conv34)
    final_conv3 = UpSampling2D((4, 4))(conv33)
    final_conv2 = UpSampling2D((2, 2))(conv32)
    final_conv1 = conv31

    conv_out = koncatenate([final_conv1, final_conv2, final_conv3, final_conv4])
    conv_out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', data_format='channels_last')(conv_out)
    conv_out = Dropout(0.2)(conv_out)
    conv_out = BatchNormalization()(conv_out)
    conv_out = Activation('relu')(conv_out)
    conv_out = Conv2D(filters=12, kernel_size=(1, 1))(conv_out)

    outmap = UpSampling2D((2, 2))(conv_out)
    model = Model(inputs=[input], outputs=[outmap])

    return model


def get_linknet(input_shape=(320, 448, 1),n_conv=64,depth=4,act='elu',n_classes=1):
    """
        https://github.com/nickhitsai/LinkNet-Keras/blob/master/linknet.py

        customiztion in progess
    """

    def _shortcut(input, residual):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[3],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)

        return add([shortcut, residual])

    def encoder_block(input_tensor, m, n):
        x = BatchNormalization()(input_tensor)
        x = Activation('elu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)

        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

        added_1 = _shortcut(input_tensor, x)

        x = BatchNormalization()(added_1)
        x = Activation('elu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)

        added_2 = _shortcut(added_1, x)

        return added_2

    def decoder_block(input_tensor, m, n):
        x = BatchNormalization()(input_tensor)
        x = Activation('elu')(x)
        x = Conv2D(filters=int(m//4), kernel_size=(1, 1))(x)

        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(filters=int(m//4), kernel_size=(3, 3), padding='same')(x)

        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = Conv2D(filters=n, kernel_size=(1, 1))(x)

        return x

    inputs = Input(shape=input_shape)

    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    encoder_1 = encoder_block(input_tensor=x, m=64, n=64)

    encoder_2 = encoder_block(input_tensor=encoder_1, m=64, n=128)

    encoder_3 = encoder_block(input_tensor=encoder_2, m=128, n=256)

    encoder_4 = encoder_block(input_tensor=encoder_3, m=256, n=512)

    decoder_4 = decoder_block(input_tensor=encoder_4, m=512, n=256)

    decoder_3_in = add([decoder_4, encoder_3])
    decoder_3_in = Activation('relu')(decoder_3_in)

    decoder_3 = decoder_block(input_tensor=decoder_3_in, m=256, n=128)

    decoder_2_in = add([decoder_3, encoder_2])
    decoder_2_in = Activation('relu')(decoder_2_in)

    decoder_2 = decoder_block(input_tensor=decoder_2_in, m=128, n=64)

    decoder_1_in = add([decoder_2, encoder_1])
    decoder_1_in = Activation('relu')(decoder_1_in)

    decoder_1 = decoder_block(input_tensor=decoder_1_in, m=64, n=64)

    x = UpSampling2D((2, 2))(decoder_1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)

    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=n_classes, kernel_size=(2, 2), padding="same")(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def get_vgg_unet(input_shape,n_classes=1,pre_train=False):

    vgg16_weights_path = "/fasthome/cvlab/RDC/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

    IMAGE_ORDERING = 'channels_last'

    assert input_shape[0]%32 == 0
    assert input_shape[1]%32 == 0

    img_input = Input(shape=input_shape)
    #decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    b1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    b2 = x
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    b3 = x
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    b4 = x
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    b5 = x

    vgg  = Model(img_input,x)
    if pre_train:
        vgg.load_weights(vgg16_weights_path,by_name=True)

    levels = [b1,b2,b3,b4,b5]

    # encoder
    o = b4

    o = ( ZeroPadding2D( (1,1)))(o)
    o = ( Conv2D(512, (3, 3), padding='valid'))(o)
    o = ( BatchNormalization())(o)

    o = (UpSampling2D( (2,2)))(o)
    o = ( koncatenate([ o ,b3],axis=3 )  )
    o = ( ZeroPadding2D( (1,1)))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid'))(o)
    o = ( BatchNormalization())(o)

    o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( koncatenate([o,b2],axis=3 ) )
    o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid') )(o)
    o = ( BatchNormalization())(o)

    o = (UpSampling2D( (2,2), ))(o)
    o = ( koncatenate([o,b1],axis=3 ) )
    o = ( ZeroPadding2D((1,1)))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'))(o)
    o = ( BatchNormalization())(o)


    o =  Conv2D( n_classes , (3, 3) , padding='same')( o )
    o = (UpSampling2D( (2,2), ))(o)
    o = Activation('sigmoid')(o)
    model = Model( img_input , o )

    return model


def get_custom_mobunetV1(input_shape = (n_rows,n_cols,1),n_conv = 8,
                       act='relu',bottle_idx=5,alpha=1.,alpha_up=1.,depth_multiplier=1):
    """
        network inspired both by unet and mobilnet implementations
    """

    conv = {}
    pool = {}
    up = {}
    kon = {}

    ## only for simplicity
    pool[0] = Input(input_shape,name='input')
    j = bottle_idx-1
    for i in range(1,2*bottle_idx):

        if i < bottle_idx:
            # encoder
            conv[i] = _depthwise_conv_block(pool[i-1],2**(i-1)*n_conv,
                                            alpha, depth_multiplier, block_id=i)
            conv[i] = _depthwise_conv_block(conv[i],2**(i-1)*n_conv,
                                            alpha, depth_multiplier, block_id=-i)
            pool[i] = _depthwise_conv_block(conv[i],2**(i-1)*n_conv,
                                            alpha, depth_multiplier, block_id=-10*i,strides=(2,2))
        elif i == bottle_idx:
            ## bottle
            conv[bottle_idx] = _depthwise_conv_block(pool[bottle_idx-1],2**(i-1)*n_conv,
                                                     alpha, depth_multiplier, block_id=i)
            conv[bottle_idx] = _depthwise_conv_block(conv[bottle_idx],2**(i-1)*n_conv,
                                                     alpha, depth_multiplier, block_id=-i)
        else:
            ## decoder
            #up[i] = UpSampling2D(size=(2, 2),name = 'up_%s' % i)(conv[i-1])
            filters = int(2**(j-1)*n_conv*alpha)
            up[i] = Conv2DTranspose(filters, (2, 2),strides=(2,2),padding='same',name='up_%s' % i)(conv[i-1])
            kon[i] = koncatenate([up[i], conv[j]],axis=3,name = 'concat_%s' % i)

            conv[i] = _depthwise_conv_block(kon[i],2**(j-1)*n_conv, alpha_up, depth_multiplier, block_id=i)

            j-=1
    outmap = Conv2D(1, (1, 1), activation='sigmoid',name = 'outmap')(conv[2*bottle_idx-1])
    model = Model(inputs=[pool[0]], outputs=[outmap])
    return model

def get_custom_sepunet(input_shape = (n_rows,n_cols,1),n_conv = 8,
                       act='relu',bottle_idx=5,alpha=1.,alpha_up=1.,depth_multiplier=1):
    """
        network inspired both by unet and mobilnet implementations
    """

    conv = {}
    pool = {}
    up = {}
    kon = {}

    ## only for simplicity
    pool[0] = Input(input_shape,name='input')
    j = bottle_idx-1
    for i in range(1,2*bottle_idx):

        if i < bottle_idx:
            # encoder
            conv[i] = SeparableConv2D(2**(i-1)*n_conv,(3,3), padding='same',
                                      name = 'sep_conv_%s0' % i, use_bias=False)(pool[i-1])


            conv[i] = SeparableConv2D(2**(i-1)*n_conv,(3,3), padding='same',
                                      name = 'sep_conv_%s1' % i, use_bias=False)(conv[i])
            pool[i] = MaxPooling2D((2,2),name='pool_%s'%i)(conv[i])

        elif i == bottle_idx:
            ## bottle
            conv[bottle_idx] = SeparableConv2D(2**(i-1)*n_conv,(3,3),  padding='same',
                                               name = 'sep_conv_%s0' % i, use_bias=False)(pool[bottle_idx-1])
            conv[bottle_idx] = SeparableConv2D(2**(i-1)*n_conv,(3,3), padding='same',
                                               name = 'sep_conv_%s1' % i, use_bias=False)(conv[bottle_idx])

        else:
            ## decoder

            filters = int(2**(j-1)*n_conv*alpha)

            up[i] = UpSampling2D(size=(2, 2),name = 'up_%s' % i)(conv[i-1])
            kon[i] = koncatenate([up[i], conv[j]],axis=3,name = 'concat_%s' % i)

            conv[i] = SeparableConv2D(2**(j-1)*n_conv,(3,3), padding='same',
                                      name = 'sep_conv_%s_up' % i, use_bias=False)(kon[i])

            j-=1
    outmap = Conv2D(1, (1, 1), activation='sigmoid',name = 'outmap')(conv[2*bottle_idx-1])
    model = Model(inputs=[pool[0]], outputs=[outmap])
    return model
