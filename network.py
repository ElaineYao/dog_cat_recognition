from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

def network():
    # Our input feature map is 150x150x3
    img_input = layers.Input(shape=(150,150,3))

    # 3 convolution modules
    # First convolution. 16 filters, 3*3. Maxpooling, 2*2
    x = layers.Conv2D(16, 3, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)

    # Second convolution. 32 filters, 3*3. Maxpooling, 2*2
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # First convolution. 64 filters, 3*3. Maxpooling, 2*2
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # two fully-connected layers
    # binary classification problem-> sigmoid activation

    # Flatten feature mao to a 1-dim tenor so we can add fully connected layers
    x = layers.Flatten()(x)

    # First fully-connected layer. RelU, 512 hidden units,independent from the output of the last layer 17*17
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout layer to prevent overfitting
    x= layers.Dropout(0.5)(x)

    # Second(Output) layer with a single node and sigmoid
    output = layers.Dense(1, activation='sigmoid')(x)

    # Create model
    model = Model(img_input, output)

    # Print the parameter of the model
    print('Parameters of the model')
    model.summary()

    # Specifications for model training
    # binary classification & sigmoid-> binary_crossentropy; rmsprop optimizer (adam, adagrad are both okay), metrics is accuracy
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])

    return model