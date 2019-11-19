from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from hyperparams import Hyperparams

H = Hyperparams()

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.clock()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch,time.clock() - self.timetaken))
        # plt.xlabel('Epoch')
        # plt.ylabel('Total time taken until an epoch in seconds')
        # plt.plot(*zip(*self.times))
        # plt.show()
    def on_train_end(self,logs = {}):
        sum_time = 0
        for j,k in self.times:
            sum_time += k
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.show()
        print("Avg epoch time : ", sum_time/H.lstm_num_epochs)