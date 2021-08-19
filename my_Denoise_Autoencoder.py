def train_model():
    Input_layers = Input(shape = (28, 28, 1))
    conv = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', activation = 'relu')(Input_layers)
    conv = MaxPooling2D((2, 2), padding = 'same')(conv)
    conv = Conv2D(8, (3, 3), padding = 'same', activation = 'relu')(conv)
    conv = MaxPooling2D((2, 2), padding = 'same')(conv)
    conv = Conv2D(8, (3, 3), padding = 'same', activation = 'relu')(conv)
    encoded = MaxPooling2D((2, 2), padding = 'same', name = 'encoder')(conv)
    
    conv = Flatten()(encoded)
    conv = Dense(32, activation = 'relu')(conv)
    conv = Dense(64, activation = 'relu')(conv)
    conv = Dropout(0.1)(conv)
    conv = Dense(128, activation = 'relu')(conv)
    conv = Dense(256, activation = 'relu')(conv)
    conv = Dense(512, activation = 'relu')(conv)
    conv = Dense(1024, activation = 'relu')(conv)
    conv = Dropout(0.25)(conv)
    conv = Dense(encoded.shape[1] * encoded.shape[2] * encoded.shape[3])(conv)
    conv = Reshape((encoded.shape[1], encoded.shape[2], encoded.shape[3]))(conv)
    print(conv.shape)
    conv = Conv2D(8, (3, 3), activation='relu', padding='same')(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = Conv2D(8, (3, 3), activation='relu', padding='same')(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = Conv2D(16, (3, 3), activation='relu')(conv)
    conv = UpSampling2D((2, 2))(conv)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv)
    autoencoder = Model(Input_layers, decoded)

    return autoencoder
    
def main(_agrv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config = config)

    train_data, test_data = np.asarray(mnist.load_data())
    train_data[0] = train_data[0].reshape(60000, 28, 28, 1)/255
    test_data[0] = test_data[0].reshape(10000, 28, 28, 1)/255
    train_data[1] = tf.keras.utils.to_categorical(train_data[1])
    test_data[1] = tf.keras.utils.to_categorical(test_data[1])
    batch_size = 128
    
    noise_factor = 0.5
    noised_train = train_data[0] + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = train_data[0].shape)
    noised_test = test_data[0] + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = test_data[0].shape)
   
    autoencoder = train_model()
    lr = 0.0008
    for _ in range(10):
        opt = tf.keras.optimizers.Adam(lr)
        progress_bar_train = tf.keras.utils.Progbar(len(train_data[0]))
        for i in range(0, len(train_data[0]), batch_size):
            with tf.GradientTape() as tape:
                train_images = train_data[0][i : min(i + batch_size, len(train_data[0]))]
                #noised_images = noised_train[i : min(i + batch_size, len(train_data[0]))]
                #pred = autoencoder(noised_images)
                #pred = tf.reshape(pred, (train_images.shape[0], 10))
                loss = tf.keras.losses.BinaryCrossentropy()(train_images, train_images)
                progress_bar_train.add(len(train_images), values = [('loss', loss)])
            gradient = tape.gradient(loss, autoencoder.trainable_variables)
            opt.apply_gradients(zip(gradient, autoencoder.trainable_variables))
        lr *= 0.9
   
