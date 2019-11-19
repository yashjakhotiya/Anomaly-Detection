import os

class Paths:

    PROJ = '/media/yash_jakhotiya/Academic/ML/Anomaly_Detection'
    dataset = os.path.join(PROJ, 'Dataset')
    cnn_encoder = os.path.join(PROJ, 'saved_models/cnn_encoder')
    lstm = os.path.join(PROJ, 'saved_models/lstm')
    log_dir = os.path.join(PROJ, 'logs')