"""
Created on Thu Jun  9 16:35:24 2022
@author: Dong.Luo
This script is building some helpers for the model
"""
import numpy as np

def test_accuacy(model,x_test,y_test):
    logits = model.predict(x_test)
    classesi = np.argmax(logits,axis=1).astype(np.uint8)
    accuracy = (y_test==classesi).sum()/classesi.size
    return accuracy,classesi

def test_accuacy_1dcnn(model,x_test,y_test, dept= 731, batch_size=8):
    test_n = x_test.shape[0]
    classes = np.full([test_n], fill_value=0, dtype=np.uint8)
    tempx = np.full([batch_size, dept, 4], fill_value=-9999, dtype = np.float32)
    tempi=0
    starti=0
    for i in range(test_n):
        tempx[tempi,:,:]=x_test[i,:,:]
        tempi=tempi+1
        if tempi>=(batch_size) or i >=(test_n -1):
            print ("process patches "+str(starti+1)+"..."+str(i+1) + "\ttempi = " + str(tempi) )
            logits = model.predict(tempx)
            classesi = np.argmax(logits,axis=1).astype(np.uint8)
            for j in range (tempi):
                jin = j + starti
                if jin > (test_n-1):
                    jin = test_n-1                
                classes[jin]=classesi[j]
            starti = i+1
            tempi=0
    accuracy = (y_test==classes).sum()/classes.size
    return accuracy,classes
        
def test_accuacy_2dcnn(model,x_test,y_test, dept= 731, batch_size=8):
    test_n = x_test.shape[0]
    classes = np.full([test_n], fill_value=0, dtype=np.uint8)
    tempx = np.full([batch_size, 32,32,dept*4], fill_value=-9999, dtype = np.float32)
    tempi=0
    starti=0
    for i in range(test_n):
        tempx[tempi,:,:,:]=x_test[i,:,:,:]
        tempi=tempi+1
        if tempi>=(batch_size) or i >=(test_n -1):
            print ("process patches "+str(starti+1)+"..."+str(i+1) + "\ttempi = " + str(tempi) )
            logits = model.predict(tempx)
            classesi = np.argmax(logits,axis=1).astype(np.uint8)
            for j in range (tempi):
                jin = j + starti
                if jin > (test_n-1):
                    jin = test_n-1                
                classes[jin]=classesi[j]
            starti = i+1
            tempi=0
    accuracy = (y_test==classes).sum()/classes.size
    return accuracy,classes

def test_accuacy_3dcnn4(model,x_test,y_test, dept= 731, batch_size=8):
    test_n = x_test.shape[0]
    classes = np.full([test_n], fill_value=0, dtype=np.uint8)
    tempx = np.full([batch_size, 4,4,dept,4], fill_value=-9999, dtype = np.float32)
    tempi=0
    starti=0
    for i in range(test_n):
        tempx[tempi,:,:,:,:]=x_test[i,:,:,:,:]
        tempi=tempi+1
        if tempi>=(batch_size) or i >=(test_n -1):
            print ("process patches "+str(starti+1)+"..."+str(i+1) + "\ttempi = " + str(tempi) )
            logits = model.predict(tempx)
            classesi = np.argmax(logits,axis=1).astype(np.uint8)
            for j in range (tempi):
                jin = j + starti
                if jin > (test_n-1):
                    jin = test_n-1                
                classes[jin]=classesi[j]
            starti = i+1
            tempi=0
    accuracy = (y_test==classes).sum()/classes.size
    return accuracy,classes
    
def test_accuacy_3dcnn(model,x_test,y_test, dept= 731, batch_size=8):
    test_n = x_test.shape[0]
    classes = np.full([test_n], fill_value=0, dtype=np.uint8)
    tempx = np.full([batch_size, 32,32,dept,4], fill_value=-9999, dtype = np.float32)
    tempi=0
    starti=0
    for i in range(test_n):
        tempx[tempi,:,:,:,:]=x_test[i,:,:,:,:]
        tempi=tempi+1
        if tempi>=(batch_size) or i >=(test_n -1):
            print ("process patches "+str(starti+1)+"..."+str(i+1) + "\ttempi = " + str(tempi) )
            logits = model.predict(tempx)
            classesi = np.argmax(logits,axis=1).astype(np.uint8)
            for j in range (tempi):
                jin = j + starti
                if jin > (test_n-1):
                    jin = test_n-1                
                classes[jin]=classesi[j]
            starti = i+1
            tempi=0
    accuracy = (y_test==classes).sum()/classes.size
    return accuracy,classes
    
def test_accuacy_cnnlstm(model,x_test,y_test, dept= 731, batch_size=8):
    test_n = x_test.shape[0]
    classes = np.full([test_n], fill_value=0, dtype=np.uint8)
    tempx = np.full([batch_size, dept, 32,32,4], fill_value=-9999, dtype = np.float32)
    tempi=0
    starti=0
    for i in range(test_n):
        tempx[tempi,:,:,:,:]=x_test[i,:,:,:,:]
        tempi=tempi+1
        if tempi>=(batch_size) or i >=(test_n -1):
            print ("process patches "+str(starti+1)+"..."+str(i+1) + "\ttempi = " + str(tempi) )
            logits = model.predict(tempx)
            classesi = np.argmax(logits,axis=1).astype(np.uint8)
            for j in range (tempi):
                jin = j + starti
                if jin > (test_n-1):
                    jin = test_n-1                
                classes[jin]=classesi[j]
            starti = i+1
            tempi=0
    accuracy = (y_test==classes).sum()/classes.size
    return accuracy,classes

def test_accuacy_lstm(model,x_test,y_test, dept= 731, batch_size=8):
    test_n = x_test.shape[0]
    classes = np.full([test_n], fill_value=0, dtype=np.uint8)
    tempx = np.full([batch_size, dept, 32*32*4], fill_value=-9999, dtype = np.float32)
    tempi=0
    starti=0
    for i in range(test_n):
        tempx[tempi,:,:]=x_test[i,:,:]
        tempi=tempi+1
        if tempi>=(batch_size) or i >=(test_n -1):
            print ("process patches "+str(starti+1)+"..."+str(i+1) + "\ttempi = " + str(tempi) )
            logits = model.predict(tempx)
            classesi = np.argmax(logits,axis=1).astype(np.uint8)
            for j in range (tempi):
                jin = j + starti
                if jin > (test_n-1):
                    jin = test_n-1                
                classes[jin]=classesi[j]
            starti = i+1
            tempi=0
    accuracy = (y_test==classes).sum()/classes.size
    return accuracy,classes
    
def prediction_accuracy (model, datasets, dataset_n):
    """
    function for predicting batch_size number of dataset each time
    args:
        model: trained model
        datasets: sliced x and y based on batchsize
        dadtaset_n: total number of datasets
    """
    classes = np.full([dataset_n], fill_value=0, dtype=np.uint8)
    gt = np.full([dataset_n], fill_value=0, dtype=np.uint8)
    index_testi = np.full(dataset_n, fill_value=False, dtype=np.bool)    
    ## doing prediction
    itt = 0
    total_n = 0
    starti = 0 
    endi = 0
    for features, labels in datasets:
        index_testi[:] = False
        starti = endi
        endi = starti+features.shape[0]
        index_testi[starti:min(endi,dataset_n)] = True
        batch_n = index_testi.sum()   # this line can get the total number of each batch
        total_n = total_n+batch_n
        itt = itt+1
        gt[index_testi] = labels[batch_n]
        logits = model.predict(features)
        # prop = tf.nn.softmax(logits).numpy()
        classes[index_testi] = np.argmax(logits,axis=1).astype(np.uint8)
    
    if total_n!=dataset_n:
        print ("\t\ttotal_n!=dataset_n")
    accuracy = 1.0*(classes==gt).sum()/classes.size
    return accuracy, classes
