from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Convolution2D, Lambda
from keras.models import Sequential

class Models:

    def getModel1(self, img_width, img_height, num_classes):
        model = Sequential([
        Convolution2D(16, 3, 3, border_mode='same', subsample=(2, 2), input_shape=(img_width, img_height, num_classes), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.2),

        Convolution2D(32, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.2),

        Convolution2D(64, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Convolution2D(128, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='tanh'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax'),
        ])
        return model


    def getModel2(self, img_width, img_height, num_classes):
        model = Sequential([
        Convolution2D(16, 3, 3, border_mode='same', subsample=(2, 2), input_shape=(img_width, img_height, num_classes), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.2),

        Convolution2D(32, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.2),

        Convolution2D(64, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Convolution2D(256, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation='tanh'),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
        ])
        return model


    def getModel3(self, img_width, img_height, num_classes):
        model = Sequential([
        Convolution2D(16, 3, 3, border_mode='same', subsample=(2, 2), input_shape=(img_width, img_height, num_classes), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.2),

        Convolution2D(32, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.2),

        Convolution2D(64, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Convolution2D(128, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='tanh'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax'),
        ])
        return model




