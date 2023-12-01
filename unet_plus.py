import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate



def conv_block(inputs, filters, kernel_size=3, activation='relu'):
    """
    Cette fonction prend en entrée une matrice, un nombre de filtres, une taille (optionelle), une fonction d'actuvation (optionelle) et retourne une convolution 2D.

    Args:
        inputs (tf.keras.layers.Input): une matrice.
        filters (int): Le nombre de filtres.
        kernel_size (int): La taille du kernel.
        activation (str): La fonction d'activation.

    Returns:
        tf.keras.layers.Conv2D: La couche de convolution.
    """
    x = Conv2D(filters, kernel_size, activation=activation, padding='same')(inputs)
    return x


def encoder_block(inputs, filters, kernel_size=3, activation='relu'):
    """
    Cette fonction prend en entrée une matrice, un nombre de filtres, une taille (optionelle), une fonction d'actuvation (optionelle) et retourne une convolution 2D et un pooling 2D.
    
    Args:
        inputs (tf.keras.layers.Input): une matrice.
        filters (int): Le nombre de filtres.
        kernel_size (int): La taille du kernel.
        activation (str): La fonction d'activation.
    Returns:
        tf.keras.layers.Conv2D: La couche de convolution.
        tf.keras.layers.MaxPooling2D: La couche de pooling.
    """
    conv = conv_block(inputs, filters, kernel_size, activation)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def decoder_block(inputs, skip_inputs, filters, kernel_size=3, activation='relu'):
    """
    Cette fonction prend en entrée une matrice, une liste de matrices, un nombre de filtres, une taille (optionelle), une fonction d'actuvation (optionelle) et retourne une convolution 2D.
    
    Args:
        inputs (tf.keras.layers.Input): une matrice.
        skip_inputs (list): une liste de matrices.
        filters (int): Le nombre de filtres.
        kernel_size (int): La taille du kernel.
        activation (str): La fonction d'activation.
    Returns:
        tf.keras.layers.Conv2D: La couche de convolution.
    """
    upsampled = UpSampling2D(size=(2, 2))(inputs)
    concatenated = Concatenate()([upsampled] + skip_inputs)
    conv = conv_block(concatenated, filters, kernel_size, activation)
    return conv


def build_unet(input_shape=(256, 256, 3), num_classes=1):
    """
    Cette fonction prend en entrée une taille d'image et un nombre de classes et retourne un modèle U-Net++.
    
    Args:
        input_shape (tuple): La taille d'image.
        num_classes (int): Le nombre de classes.
    Returns:
        tf.keras.Model: Le modèle U-Net++.
    """
    inputs = Input(input_shape)
    
    # Encoder
    conv1, pool1 = encoder_block(inputs, 64)
    conv2, pool2 = encoder_block(pool1, 128)
    conv3, pool3 = encoder_block(pool2, 256)
    conv4, pool4 = encoder_block(pool3, 512)
    
    conv_centre = conv_block(pool4, 1024)
    
    # Skip connections
    skip1_1 = decoder_block(conv2, [conv1], 64)
    skip2_1 = decoder_block(conv3, [conv2], 128)
    skip3_1 = decoder_block(conv4, [conv3], 256)
    skip1_2 = decoder_block(skip2_1, [conv1, skip1_1], 64)
    skip2_2 = decoder_block(skip3_1, [conv2, skip2_1], 128)
    skip1_3 = decoder_block(skip2_2, [conv1, skip1_1, skip1_2], 64)
    
    # # Decoder
    upconv4 = decoder_block(conv_centre, [conv4], 512)
    upconv3 = decoder_block(upconv4, [conv3, skip3_1], 256)
    upconv2 = decoder_block(upconv3, [conv2, skip2_1, skip2_2], 128)
    upconv1 = decoder_block(upconv2, [conv1, skip1_1, skip1_2, skip1_3], 64)
    
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(upconv1)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = build_unet()
    model.summary()