import tensorflow as tf

def residual_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.add([inputs, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def get_model():
    inputs = tf.keras.layers.Input(shape=(None, None, 3))
    batch_size = tf.shape(inputs)[0]
    
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
    
    res1 = residual_block(conv3, 128)
    res2 = residual_block(res1, 128)
    res3 = residual_block(res2, 128)
    res4 = residual_block(res3, 128)
    res5 = residual_block(res4, 128)
    
    deconv1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(res5)
    deconv2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(deconv1)
    
    outputs = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(deconv2)
    outputs=tf.keras.layers.add([inputs, outputs])

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
