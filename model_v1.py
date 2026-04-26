import tensorflow as tf

def model_v1():
    # Defining the Input Shape
    inputs = tf.keras.layers.Input(shape=(3, 72, 72))

    #Layer1 Convolution with max pooling
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_first')(conv1)

    #Layer2 Convolution with max pooling
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_first')(conv2)

    #Layer3 Convolution 
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(pool2)
    #Global Average Pooling to collapse spatial dimensions
    global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(conv3)
    #Dense Layers
    dense1 = tf.keras.layers.Dense(64, activation='relu')(global_avg_pool)
    dense2 = tf.keras.layers.Dense(10)(dense1)
    dense3 = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

    return tf.keras.Model(inputs=inputs, outputs=dense3)



