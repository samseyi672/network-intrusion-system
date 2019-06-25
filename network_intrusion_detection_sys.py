# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:13:55 2019

@author: ZBOOK
"""
'''
Created on Apr 26, 2018

@author: administrator
'''

#from tensorflow.contrib.learn.python import learn
import  tensorflow as tf 
import  pandas as  pd 
import  numpy as  np
import matplotlib.pyplot as  plt
from sklearn.cross_validation import train_test_split
import tensorflow.contrib.learn as  learn
from sklearn.metrics import classification_report
from sklearn.metrics import  confusion_matrix 
from sklearn.preprocessing import  StandardScaler
from tensorflow.contrib.layers.python.layers import feature_column
from sklearn import  preprocessing
from cProfile import label
#from tensorflow.python.estimator.canned.dnn import DNNClassifier

learning_rate   = 0.001
training_epochs  =15  # no of traing cycles
batch_size  = 32
n_classes   = 10 # no of poutput
#setting the neural network
#input layer
n_input  = 42
#setting  hidden layer
n_hidden_1  =  256
n_hidden_2   = 256
n_samples=None
def readMyDataSet():
    print('reading the  data set ..........')
    df = pd.read_csv('C:\\Users\\Administrator.ADSVR\\Downloads\\KDDCup99\\KDDCup99.csv')
    #making  use  of just 10% of  the dataset 
    df = df.sample(frac=0.1, replace=False) 
    #removing rows  with missing values 
    df.dropna(inplace=True,axis=1)
    df = df.dropna(how="any", axis=0)
    print('printing out head  of  dataframe ',df.head())
    print('describing the  dataframe ',df.describe())
    return df  # return the dataset  as a dataframe
 
def plotTheImageOfTheDataset(df):
    plt.imshow(df[0:2],cmap='Grey')
    
    
#creating the  recurrent neural network
#using a tensorflow  recurrent  neural network 
#this  is one to  create rnn but  i  did not  use this
def multilayerperceptron(x ,weights,biases):# this  is your   model
    print('inside  the multilayer perceptron')
    global learning_rate 
    global training_epochs
    global batch_size
    global n_classes  # no of poutput
    global n_samples
    n_samples   = readMyDataSet()# calling the function
    #setting the neural network
    #input layer
    n_input
    #setting  hidden layer
    n_hidden_1
    n_hidden_2
    
    '''
    x:placeholder for data input
    weights: dict of weights
    biases: dict of bias values
    ''' 
    #first Hidden Layer with RELU Activation
    #X*W+ B
    layer_1  = tf.add(tf.matmul(x, weights['h1']),biases['b1'])
    #func(X*W+B)
    layer_1   =  tf.nn.relu(layer_1)
     #second hidden layer
    layer_2  = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2   =  tf.nn.relu(layer_2)
    #last  output layer
    out_layer   = tf.matmul(layer_2 ,weights['out'])+ biases['out']
    return out_layer

def defineWeightsOfNeuron():
   #weights as a dictionary
   #initializing weight randomly
   weights={
       'hi':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
       'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),                
       'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
        }
   biases={
       'b1':tf.Variable(tf.random_normal([n_hidden_1])),
       'b2':tf.Variable(tf.random_normal([n_hidden_2])),                
       'out':tf.Variable(tf.random_normal([n_classes]))  
       }
   return weights , biases # returning both weights and biases

#tensorflow  placeholder
def setTensorPlaceHolder():
    x = tf.placeholder('float',[None,n_input])
    y= tf.placeholder('float',[None,n_classes])
    return x,y

#encode text values  to indexes
#(i.e [1][2][3] for red green blue)
def encode_text_index(df ,name):
    le  = preprocessing.LabelEncoder()
    df[name] =  le.fit_transform(df[name])
    return le.classes_

# convert a pandas dataframe to the x,y inputs that tensorflow needs
def to_xy(df):
# standadizing the data
     print('standizing the data ......')
     scaler  = StandardScaler()
     #fitting it  the fesatures
     encode_text_index(df, 'label')
     print('label encoded') 
     #scaler.fit(df.drop('label',axis=1))
     #print('transforming it to  a scaled features ')
     #scaled_features   = scaler.fit_transform(df.drop('label'),axis=1)
    #converting it back to  a dataframe
    #df_feat  = pd.DataFrame(scaled_features,columns=df.columns[:-1])
     #print(df_feat.head())# to  check if the  scaling work    
     x  = df
     y=df['label']
     print('resetting  x and  y to numpy array for tensorflow')  
     x  = x.as_matrix()
     y =  y.as_matrix()
     return x ,y
    
    
# i did  not  this  to make  predictions
def makePrediction():
    x,y =setTensorPlaceHolder()
    weights ,biases  = defineWeightsOfNeuron()
    prediction  = multilayerperceptron(x, weights, biases)
    return x,y ,prediction

def setCostfunctionAndOptimizer(prediction=None):
    global learning_rate
    x,y = to_xy(readMyDataSet())
    print('prepared x and y')
    #cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,x,y))
    #optimizer   = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #print out   everything 
    #print('prediction ' ,prediction ,' cost function ',
          #cost,' optimizer ' ,optimizer)
    return x,y



def input_fn(data_set,LABEL,FEATURES):
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels
 
def train_input_fn(y):
    #x,y = to_xy(readMyDataSet())[0:5]
    #feature_cols = tf.constant(x)
    labels = tf.constant(y)
    session  = tf.Session()
    return session.run(labels)
    #return labels

# using the  tensor.contrib to from the model     
def model(x,y):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)
    #hidden_units=[10,20,10] mean 10,20,10 nodes  per  layer
    #this is rnn that  i use
    #get_train_inputs(x_train, y_train)
    print('got all  x and y ')
    FEATURES=['duration','protocol_type','service','flag' ,'src_bytes']
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
    LABEL='label'
    #classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)
    my_feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)] 
    classifier = learn.DNNRegressor(feature_columns=my_feature_columns,hidden_units=[10,20,10])
    print('training the  model')
    #classifier.fit(input_fn=train_input_fn(x_train.astype(str)), steps=10)
    print('the classifier  is  working...')
    classifier.fit(x_train,y_train,steps=10)
    print('the classifier  is  working...')
    classifier.evaluate(y_test.astype(bytes))
    
    predictions = classifier.predict(x_test)
    print('predictions.,,,,,,', 'and the type ',type(predictions))
    print(predictions)
    print('classification report ')
    
    print(classification_report(y_test ,predictions))
    print(confusion_matrix(y_test ,predictions))
    tf_confusion_metrics_npr(classifier,n_classes,tf.Session(),)
    return predictions

#to  calculate  fpr  and  tpr
def tf_confusion_metrics_npr(model, actual_classes, session, feed_dict):
  predictions = tf.argmax(model, 1)
  actuals = tf.argmax(actual_classes, 1)
 
  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)
 
  tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )
 
  tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )
 
  fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )
 
  fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )
 
  tp, tn, fp, fn = \
    session.run(
      [tp_op, tn_op, fp_op, fn_op], 
      feed_dict
    )
 
  tpr = float(tp)/(float(tp) + float(fn))
  fpr = float(fp)/(float(tp) + float(fn))
 
  accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
  return tpr , fpr 
# training the  model now 
def trainTheModel():
    global training_epochs
    print('training the  model.....')
    x,y = setCostfunctionAndOptimizer()
    # Create a test/train split.  25% test
    # Split into train/test
    prediction  =  model(x,y)

trainTheModel()




# #another  version
# prediction=tf.nn.softmax(tf.matmul(X,w)+b)
# pred_pos= prediction.eval(feed_dict={X: x_pos})
# pred_neg= prediction.eval(feed_dict={X: x_neg})
# tpr=[]
# fpr=[]
# for j in range(100):
#     pos=0
#     neg=0
#     n=j/100.
#     for i in range(0,len(pred_pos)):
#             if(pred_pos[i,1]>=n):
#                 pos+=1
#             if(pred_neg[i,1]>=n):
#                 neg+=1
#     tpr.append(pos/len(x_pos))
#     fpr.append(neg/len(x_neg))
# 
# f= open('output.txt','wb')
# arr=np.array([fpr,tpr])
# arr=arr.T          
# np.savetxt(f,arr,fmt=['%e','%e'])    
# f.close()
# 
# 
# # one sample  for fp
# For the multi-class case, everything you need can be found from the confusion matrix. For example, if your confusion matrix looks like this:
# 
# confusion matrix
# 
# Then what you're looking for, per class, can be found like this:
# 
# overlay
# 
# Using pandas/numpy, you can do this for all classes at once like so:
# 
# FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
# FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
# TP = np.diag(confusion_matrix)
# TN = confusion_matrix.values.sum() - (FP + FN + TP)
# 
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP) 
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)
# 
# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
# 
# 
# 
# #edit this code  for  tpr  and fpr 
# # from https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data

#   recall = tpr
#   precision = float(tp)/(float(tp) + float(fp))
#   
#   f1_score = (2 * (precision * recall)) / (precision + recall)
#   
#   print 'Precision = ', precision
#   print 'Recall = ', recall
#   print 'F1 Score = ', f1_score
# print 'Accuracy = ', accuracy
#     





















































