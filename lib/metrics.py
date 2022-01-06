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
         
smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def jaccard(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def jacard_dice(yp, Y_test):
    
    #yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    #yp = np.round(yp,0)

    jacard = 0
    dice = 0
    smooth = 0.0
    for i in range(len(Y_test)):
        flat_pred = K.flatten(Y_test[i])
        flat_label = K.flatten(yp[i])
        
        intersection_i = K.sum(flat_label * flat_pred)
        union_i = K.sum( flat_label + flat_pred - flat_label * flat_pred)
        
        dice_i = (2. * intersection_i + smooth) / (K.sum(flat_label) + K.sum(flat_pred) + smooth)
        jacard_i = intersection_i / union_i
        
        jacard += jacard_i
        dice += dice_i

    jacard /= len(Y_test)
    dice /= len(Y_test)
    print(jacard.numpy())
    print(dice.numpy())
    
    return jacard.numpy(), dice.numpy()


def evaluateModel(yp, Y_test):
    #yp = model.predict(x=X_test, verbose=1)
    #yp = np.round(yp,0)
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
    

    #dice, jacard = jacard_dice(yp, Y_test)
    
    return r2, r1, r3
    #prec, rec, f1, iou = evaluate_segmentation(yp, Y_test)
    
    
def testModel(model, X_test, Y_test, batchSize, experiment_name):
    result_folder = "../results/" + experiment_name
    create_dir(result_folder)
    
    csv_filename = "../results/" + 'test_results.csv'
    create_csv_file(csv_filename)

    jacard = 0
    dice = 0
    results = []
    record_results = pd.DataFrame()
    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)
    
    for i in range(len(Y_test)):

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

        jacard = (np.sum(intersection)/np.sum(union))  
        plt.suptitle('Jacard Index '+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))

        plt.savefig(result_folder + "/" + str(i)+'.png',format='png')
        plt.close()
        
    
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection)/np.sum(union))  
        dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))

    jacard /= len(Y_test)
    dice /= len(Y_test)
    
    acc, binary_acc, mean_iou = evaluateModel(yp,Y_test)
    results.append(experiment_name)
    results.append(acc)
    results.append(binary_acc)
    results.append(mean_iou)
    #results.append(jacc)
    #results.append(dice_coef)
    results.append(jacard)
    results.append(dice)
    
    with open(csv_filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)
        f_object.close()
    
    print('Test Jacard Index : '+str(jacard))
    print('Test Dice Coefficient : '+str(dice))
    
    
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
        