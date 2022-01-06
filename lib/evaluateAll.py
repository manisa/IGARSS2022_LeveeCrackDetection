import os
from pathlib import Path
from csv import writer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def create_csv_file(path_to_file):
    path = Path(path_to_file)
    if not path.is_file():
        list_names = ['experiment_name','accuracy', 'binary_accuracy', 'mean_iou', 'jaccard', 'dice_coeff']
        with open(path, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(list_names)
            f_object.close()


def evaluateModel(yp, Y_test):
    flat_pred = K.flatten(Y_test)
    flat_label = K.flatten(yp)

    binaryacc = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)
    acc = tf.keras.metrics.Accuracy()
    #auc = tf.keras.metrics.AUC()
    miou = tf.keras.metrics.MeanIoU(num_classes=2)
    
    r1 = binaryacc.update_state(flat_label,flat_pred)
    r1 = binaryacc.result().numpy()
    
    r2 = acc.update_state(flat_label,flat_pred)
    r2 = acc.result().numpy()
    
    r3 = miou.update_state(flat_label,flat_pred)
    r3 = miou.result().numpy()
    
    return r2, r1, r3
    
    
def foldTestModel(model, X_test, Y_test, batchSize, experiment_name, fold_number):
    result_folder = "../results/" + experiment_name + "_" + str(fold_number)
    create_dir(result_folder)
    
    csv_filename = "../results/" + 'test_results.csv'
    create_csv_file(csv_filename)

    jacard = 0
    dice = 0
    results = []
    record_results = pd.DataFrame()
    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)
    smooth = 1e-15
    for i in range(10):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jaccard = ((np.sum(intersection) + smooth)/(np.sum(union) + smooth))  
        plt.suptitle('F1 Score computed as: '+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +' = '+str(jaccard))

        plt.savefig(result_folder + "/" + str(i)+'.png',format='png')
        plt.close()
        
    
    jaccard = 0.0
    dice = 0.0
    
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jaccard += ((np.sum(intersection) + smooth)/(np.sum(union) + smooth))  
        
        dice += (2. * np.sum(intersection)+smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)

    jaccard /= len(Y_test)
    dice /= len(Y_test)
    
    acc, binary_acc, mean_iou = evaluateModel(yp,Y_test)
    results.append(experiment_name)
    results.append(acc)
    results.append(binary_acc)
    results.append(mean_iou)
    results.append(jaccard)
    results.append(dice)
    
    with open(csv_filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)
        f_object.close()
    
    print('Test Jacard Index : '+str(jaccard))
    print('Test Dice Coefficient : '+str(dice))
    return results
    
def saveResultsOnly(model, X_test, batchSize, experiment_name):
    result_folder = "../Outputs/" + experiment_name
    create_dir(result_folder)
    
    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)
    
    for i in range(len(X_test)):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
    
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        plt.savefig(result_folder + "/" + str(i)+'.png',format='png')
        plt.close()
        
