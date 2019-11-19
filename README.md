# Anomaly-Detection

This project is trained on the [UCSD dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm). 
The code has been kept in modular format, with the following modules -
1. `data_loader.py` - Contains two `tf.keras.utils.Sequence` classes - `CNN_train_data_loader` and `CNN_test_data_loader` to
load batches of images and labels
2. `paths.py` and `hyperparams.py` - Contains paths and hyperparams needed for the model
3. `logger.py` - Logger callback
4. `timer.py`- Timer callback
5. `train.py` - Entry point for the code; starts the training sequence
6. `model.py` - Contains two models - a CNN Autoencoder and a 2-layered LSTM

## CNN Autoencoder

A CNN Autoencoder is trained to learn latent space representation of frame images

## Stacked LSTM

A 2-layered LSTM is trained to predict if the sequence of frames contains an anomaly. 
Latent space representation of each frame is passed to the stacked LSTM.

