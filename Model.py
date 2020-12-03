# Importig required Libaries
from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Flatten, Cropping2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda
from keras.models import model_from_json


def Pattern_Recognion_Model_API(X_train,y_train):
    '''
    Pattern Recognition model with the use Functional API method consists of 24 Sequential layers 
        6 Convolutional layers followed by relu activation layer.
        5 Fully connected layers followed bye relu activation layer.
    Convolutional layers plays important role in segregating all lane curve extractions and curvation associated information.
    Fully connected layers plays important role in reducing network size layer by layer in extracting curvature of lanes in taking the steering angle prediction
    
    Loss
    ----
    mean squared error loss is considered in optimizing the model performance
    Optimizer
    --------
    Adam optimizer is considered in changing the learning rate to converging Neural network in getting high performance in prediction
    default learning rate of 0.9 to 0.999 increment of optimization with the step of 0.001 is considered
    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    
    '''

    X_input = Input(shape=X_train.shape, name='img_in')
    #X =  Cropping2D(cropping=((70, 25), (0, 0)))(X_input)
    #X = Lambda(lambda image: ktf.image.resize_images(image, (80, 200)))(X)
    X = Lambda(lambda x: (x / 255.0) - 0.5)(X_input)
    X = Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=6, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)

    X = Activation('relu')(X)
    # X = Conv2D(filters=6, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
    #                  kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    # X = Activation('relu')(X)
    X = Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=32,kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = Activation('relu')(X)
    # Fully connected
    X = Flatten()(X)
    # model.add(Dropout(0.35))
    X = Dense(units=1164)(X)
    X = Activation('relu')(X)
    X = Dense(units=100)(X)
    X = Activation('relu')(X)
    X = Dense(units=50)(X)
    X = Activation('relu')(X)
    X = Dense(units=10)(X)
    X = Activation('relu')(X)
    X = Dense(units=1)(X)
    model=Model(inputs=X_input, outputs=X, name='Convolve')
    model.compile(optimizer='adam',loss='mean_squared_error',metrics = ['mse'])
    return model




def PatternRecognitionModel(input_shape):
    '''
    Pattern Recognition model consists of 24 Sequential layers 
        6 Convolutional layers followed by relu activation layer.
        5 Fully connected layers followed bye relu activation layer.
    Convolutional layers plays important role in segregating all lane curve extractions and curvation associated information.
    Fully connected layers plays important role in reducing network size layer by layer in extracting curvature of lanes in taking the steering angle prediction
    
    Loss
    ----
    mean squared error loss is considered in optimizing the model performance
    Optimizer
    --------
    Adam optimizer is considered in changing the learning rate to converging Neural network in getting high performance in prediction
    default learning rate of 0.9 to 0.999 increment of optimization with the step of 0.001 is considered
    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    
    '''
    # Model
    model = Sequential()
# Convolutional
    model.add(Cropping2D(cropping=((5, 5), (0, 0)), input_shape=input_shape))
#     Lambda(lambda image: ktf.image.resize_images(image, (80, 200)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1), use_bias=True,
                     kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', dilation_rate=(1, 1),use_bias=True, 
                     kernel_initializer='glorot_uniform',bias_initializer='zeros'))
    model.add(Activation('relu'))
    # Fully connected
    model.add(Flatten())
    model.add(Dense(units=1164))
    model.add(Activation('relu'))
    model.add(Dense(units=100))
    model.add(Activation('relu'))
    model.add(Dense(units=50))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])
    return model



