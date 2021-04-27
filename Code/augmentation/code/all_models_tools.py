import os
import pathlib
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from glob import glob
from tqdm import tqdm
tqdm().pandas()
from sklearn import preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model



def prepare_dataset(path,label):
    x_train=[]
    y_train=[]
    all_images_path=glob(path+'/*.jpg')
    for img_path in tqdm(all_images_path) :
            img=load_img(img_path, target_size=(150,150))
            img=img_to_array(img)
            img=img/255.0
            x_train.append(img)
            y_train.append(label)
    return x_train,y_train



def plot_learning_curve(history, name):
    '''
    Function to plot the accuracy curve
    @param history: (history object) containing all the relevant information about the training
    @param name: (string) name of the model
    '''
    # extract informations
    acc     = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss    = history.history["loss"]
    val_loss= history.history["val_loss"]
    epochs  = range(1, len(acc)+1)
    # plot accuracy
    plt.plot(epochs, acc, 'b-o', label='Training loss')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(name+"_accuracy.png") # save the picture, put into comment if you don't want
    plt.figure()
    # plot losses
    plt.plot(epochs, loss, 'b-o', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(name+"_losses.png") # save the picture, put into comment if you don't want
    plt.show()

# Functions to extract the true, false positive and true false negative
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


score_metrics = {'acc': accuracy_score,
               'balanced_accuracy': balanced_accuracy_score,
               'prec': precision_score,
               'recall': recall_score,
               'f1-score': f1_score,
               'tp': tp, 'tn': tn,
               'fp': fp, 'fn': fn,
               'cohens_kappa':cohen_kappa_score,
               'matthews_corrcoef':matthews_corrcoef,
               "roc_auc":roc_auc_score}

def metrics_evaluation(y_test, y_pred, dic_score=score_metrics):
    '''
    @param y_test: (numpy array) true labels
    @param y_pred: (numpy array) predicted labels
    @param dic_score: (dict) dictionary containing the metrics for the evaluation
    '''

    scorer = {}
    for i in dic_score.keys():
        scorer[i] = None
        # metrics can be used for binary and multiclass classification problem
        if len(set(y_test)) > 2:
            if i in ["prec", "recall", "f1-score"]:

                scorer[i] = round(dic_score[i](y_test, np.argmax(y_pred, axis=1), average='weighted'),
                                  4)  # make each function scorer

            elif i == "roc_auc":
                scorer[i] = round(
                    dic_score[i](to_categorical(y_test), y_pred, average='macro', multi_class="ovo"),
                    4)  # make each function scorer
            else:
                scorer[i] = dic_score[i](y_test, np.argmax(y_pred, axis=1))  # make each function scorer

        else:
            scorer[i] = round(100 * dic_score[i](y_test, y_pred), 2)

    return scorer


def pretrained_model_classification(x_train, y_train, x_valid, y_valid, x_test, y_test, _model, batch_size=32,
                                    epochs_num=100, patience=3, num_classes = 5):
    '''
    This function create a new model using transfer learning, train the model and evaluate it.
    @param x_train: (numpy array, matrix) matrix containing pixels values of the train set
    @param y_train: (numpy array) list of label (int values) for the train set
    @param x_valid: (numpy array, matrix) matrix containing pixels values of the validation set
    @param y_valid: (numpy array) list of label (int values) for the validation set
    @param x_test: (numpy array, matrix) matrix containing pixels values of the test set
    @param y_test: (numpy array) list of label (int values) for the test set
    @param _model: (model) pretrained model
    @param batch_size: (int) number of images for each batch default 32
    @param epochs_num: (int) number of epochs for the fit() default 100
    @param patience: (int) number of iteration without learning to break the training default 3
    @return history: (history object) containing the information of the training process
    @return results: (dataframe) pandas dataframe containing the metrics of the model evaluated on the test set
    '''
    # create an instance of the pretrained model
    base_model = _model
    # freeze the layers of the pretrained model
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(5, activation='softmax')(x)
    # create a new model with pretrained + last new layers
    model_final = Model(base_model.input, x)
    # compile
    model_final.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    # fit the model with early stopping
    es = tf.keras.callbacks.EarlyStopping(patience=patience)
    history = model_final.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid), callbacks=[es], epochs=epochs_num, batch_size=batch_size)

    # predict the test
    y_pred = model_final.predict(x_test)
    return history

    '''
    # evaluate the results on the test set
    metrics = metrics_evaluation(y_test, y_pred, dic_score=score_metrics)
    results = pd.DataFrame(metrics.values(), index=metrics.keys()).T

    return history, results
    '''