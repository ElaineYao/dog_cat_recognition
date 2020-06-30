from utils import *
from network import *

def train():
    model = network()
    (train_generator, validation_generator) = preprocessing()
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=2,
            validation_data=validation_generator,
            validation_steps=50,
            verbose=2
    )
    plot_acc_loss(history)

if __name__ == '__main__':
    train()