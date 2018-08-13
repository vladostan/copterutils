from keras import backend as K
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred,smooth = 1.0):
    """
        Dice функция потерь
        y_true - тензор с разметкой 
        y_pred - тензор с предсказаниями
    """
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true,y_pred):
    """
        
    """
    return 1 - dice_coef(y_true,y_pred)

def log_dice_crossentopy_loss(y_true,y_pred,W=[1.,1.]):
    """
     взвешенная (?) сумма кросс энтропии и логарифма  dice
    """
    b = binary_crossentropy( K.clip(y_true, K.epsilon(), 1.),K.clip(y_pred, K.epsilon(), 1.))
    d = dice_coef(y_true,y_pred)
    d = K.log(d)
    return 1 - W[0]*d + W[1]*b 


def dice_crossentopy_loss(y_true,y_pred,W=[1.,1.]):
    """
     взвешенная (?) сумма кросс энтропии и dice
    """
    b = binary_crossentropy( K.clip(y_true, K.epsilon(), 1.),K.clip(y_pred, K.epsilon(), 1.))
    d = dice_coef(y_true,y_pred)
    return 1 - W[0]*d + W[1]*b 

def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))

    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


def f1_score(y_true, y_pred):
    """
    
    """
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))

def dice_square(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2 * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

def dice_square_loss(y_true, y_pred):
    return 1 - dice_square(y_true, y_pred)
