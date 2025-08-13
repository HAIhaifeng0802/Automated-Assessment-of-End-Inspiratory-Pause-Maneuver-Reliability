import datetime
import pandas as pd
from tqdm import tqdm
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from contextlib import redirect_stdout  
from dataset_smm import ablation_dataset
from utils.utils_generic import custom_augment, pre_process
from models.utils_loss import plot_progress, plot_progress_multi_task
from models.utils_model import get_projection, get_linear_classifier
from models.cnn_small import SmallCNN
from utils.utils_metris import create_folder, plot_confusion_matrix, get_metris, get_metrics_new, save_json, save_table, save_table2
from quant import converting_model_lce, converting_model_tf
from convert import h5_to_pb_online
# import larq as lq
import warnings
warnings.filterwarnings("ignore")
#%% Algorithm hyperparameters
now = datetime.datetime.now()
exp = now.strftime("%Y-%m-%d-%H-%M-%S")

save_his_path = 'trained_model'

num_epochs = 1000
batch_size = 64

seed = 13
maxlen = 300
n_classes = 3

ratio = 1
# n_splits = 10
n_splits = 5

labels = ['JTLX_Standard']

# case = 'float32'
# # case = '1bit'

# if case == 'float32':
x, y, x_index = ablation_dataset(
                                    maxlen=maxlen,
                                    savepath=os.path.join(os.getcwd(), exp, 'data'),
                                    norm='z-score'
                                    )
print(len(x))

kfold = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

count = 0
train_result = {}

save_dir = os.path.join(os.getcwd(), exp)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
for train_index, test_index in kfold.split(x, y):
    
    x_train, x_test, x_train_id, x_test_id, y_tra, y_te = np.array(x)[train_index], np.array(x)[test_index],np.array(x_index)[train_index], np.array(x_index)[test_index], np.array(y)[train_index], np.array(y)[test_index]
        
    # y_train = [np.where(np.array(y_tra)!=1, 0, 1), np.where(np.array(y_tra)!=2, 0, 1), np.where(np.array(y_tra)!=3, 0, 1), np.where(np.array(y_tra)!=4, 0, 1), np.array([1 if x == 5 else 2 if x == 6 else 0 for x in y_tra])]
    # y_test = [np.where(np.array(y_te)!=1, 0, 1), np.where(np.array(y_te)!=2, 0, 1), np.where(np.array(y_te)!=3, 0, 1), np.where(np.array(y_te)!=4, 0, 1), np.array([1 if x == 5 else 2 if x == 6 else 0 for x in y_te])]

    y_train = y_tra    
    y_test = y_te

    x_train = np.reshape(x_train, (-1, maxlen, 3))
    x_test = np.reshape(x_test, (-1, maxlen, 3))
    
    
#%% create linear model

    tf.keras.backend.clear_session()
    model = SmallCNN(maxlen=maxlen, 
                    active_learner=False,
                    include_top=True, 
                    classify=False,
                    classes=n_classes,
                    model_name='cnn1d_multi_task',
                    )
    # model.summary()
    # lq.models.summary(model)
    # if not os.path.exists(os.path.join(save_dir, 'model_summary.txt')):
        # with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
            # with redirect_stdout(f):
                #model.summary()
                # lq.models.summary(model)
    plot_model(model, to_file=os.path.join(save_dir, '1dmodel.png'), show_shapes=True)
    
    rootpath = os.path.join(save_dir, save_his_path, str(count))
    if not os.path.isdir(rootpath):
        os.makedirs(rootpath)
    
    #model_name = 'model_{epoch:03d}_{val_DelayedCycling_accuracy:.03f}_{val_PrematureCycling_accuracy:.03f}_{val_DoubleTrig_accuracy:.03f}_{val_InefTrig_accuracy:.03f}_{val_loss:.03f}.h5'
    model_name = 'model_{epoch:03d}_{val_accuracy:.03f}_{val_loss:.03f}.h5'
    model_path = os.path.join(rootpath, model_name)
    
    model_checkpoint_callback = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    early_stopper_callback = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, cooldown=5, min_lr=1e-9)
    
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
                  )
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        shuffle=True,
                        verbose=1,
                        validation_split=0.3,
                        # validation_data=(x_test, y_test),
                        callbacks=[early_stopper_callback, 
                                    reduce_lr_callback,
                                    model_checkpoint_callback, 
                                    ]
                        )
    
    create_folder(save_dir, count) 
    
    savepath = os.path.join(save_dir, 'predict', str(count))
    
    plot_progress(history, savepath=savepath, save_history=os.path.join(savepath, 'history.npy'))
    # (history, 
    #                           save_dir=savepath, 
    #                           save_history=os.path.join(savepath, 'history.npy'))
    
    #model.save_weights(os.path.join(rootpath, "model.h5"))
    
    #%% Validation
    
    proba = model.predict(x_test, batch_size=batch_size)
    # loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', loss)
    # print('Test accuracy:', accuracy)
    pred_label = []
    # print(len(proba))
    for i in range(len(proba)):
        # for j in range(5):
        idx = np.argmax(proba[i])
        # print(proba[i])
        # print(idx)

        if idx == 0:
            pred_label.append(0)
        elif idx == 1:
            pred_label.append(1)
        else:
            pred_label.append(2)
            
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_test, pred_label, labels=[0, 1, 2])
    plot_confusion_matrix(confusion, savepath, labels[0], ['Non-EIPM','Unreliable', 'Candidate'])
            
    tmp = {}

    tmp.update({labels[0] : get_metris(y_test, pred_label, proba, classes=3)})
    # tmp.update({labels[1] : get_metrics(y_test, pred_label, proba, classes=3)})

    train_result.update({count : tmp})
    
    # #%%  Converting the model
    #     # Convert our Keras model to a TFLite flatbuffer file
    # if case == 'float32':
    #     converting_model_tf(rootpath, model)

    # elif case == '1bit':
    #     converting_model_lce(rootpath, model)
    
    # # Convert our Keras model to a pb file
    # h5_to_pb_online(model, pb_save_path=rootpath, pb_save_name='full_precision_model')
    
    count = count + 1
#     if count==1:
#         break
# n_splits = 1
save_table(save_dir, n_splits, train_result)
save_table2(save_dir, n_splits, train_result)
save_json(save_dir, train_result)
