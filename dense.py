import numpy as np

from layers import *
from losses import CrossEntropyLoss
from activations import ReLU
from net import Net
from Utils import *
import math 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from evaluation_matrix import *
# GRADED FUNCTION: random_mini_batches
  
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X.T, mini_batch_Y.T)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X.T, mini_batch_Y.T)
        mini_batches.append(mini_batch)
    
    return mini_batches




X_train,y_train,_=load_data("mnist_train.csv")
X_test,y_test,_ = load_data("mnist_test.csv")
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train.reshape(-1,X_train.shape[1]), X_test.reshape(-1,X_test.shape[1])
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
X_dev,y_dev=X_train[55000:60000,:],y_train[55000:60000,:]
X_train,y_train=X_train[0:55000,:],y_train[0:55000,:]
net = Net(layers=[Linear(X_train.shape[1], 512,mode=1,layerNo=1),ReLU(), Linear(512, 512,mode=1,layerNo=2),ReLU(), Linear(512, 10,mode=1,layerNo=3)],
          loss=CrossEntropyLoss())


# reshaping
#X_train, X_test = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)
#y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
# normalizing and scaling data
X_train, X_test,X_dev = X_train.astype('float32')/255, X_test.astype('float32')/255,X_dev.astype('float32')/255
y_train, y_test,y_dev=y_train.astype('int8'), y_test.astype('int8'),y_dev.astype('int8')
n_epochs = 0
mini_batches = random_mini_batches(X_train.T, y_train.T)  
i=0  
accuracy_list = []
for epoch_idx in range(n_epochs):
    accuracy = 0
    dev_accuracy = 0
    for minibatch in mini_batches:
        (minibatch_X, minibatch_Y) = minibatch
        out = net(minibatch_X)
        loss = net.loss(out, minibatch_Y)
        net.backward()
        net.update_weights(lr=0.1,i=i ,layer_type = 'linear')
    if epoch_idx==n_epochs-1 :
        i=1
        net.update_weights(lr=0.1,i=i,layer_type = 'linear')
    out = net(X_train)
    preds = np.argmax(out, axis=1).reshape(-1, 1)
    accuracy = 100*(preds == y_train).sum() / 55000
    out = net(X_dev)
    preds_dev = np.argmax(out, axis=1).reshape(-1, 1)
    dev_accuracy = 100*(preds_dev == y_dev).sum() / 5000    
    print("Epoch no. %d loss =  %2f4 \t train_accuracy = %d %%" % (epoch_idx + 1, loss, accuracy))
    print('dev_accuracy = %d %%' % (dev_accuracy))
    accuracy_list.append([accuracy, dev_accuracy])
accuracy_list = np.array(accuracy_list).T

out = net(X_train)
preds_train = np.argmax(out, axis=1).reshape(-1, 1)
micro_f1 = micro_F1_SCORE(y_train,preds_train)
print("micro F1 score for training = micro precision = micro recall = "+str(micro_f1)+'\n')

hot_form_y=hot_form(y_train,10)
hot_form_pred=hot_form(preds_train,10)

f1_score_arr, precision_arr, recall_arr =f1_score_labels(hot_form_y ,hot_form_pred)
print("f1 score for train = "+str(f1_score_arr)+'\n')
print("precision for train = "+str(precision_arr)+'\n')
print("recall for train = "+str(recall_arr)+'\n')
macro_f1_score_train,macro_precision_train,macro_recall_train = macro_f1_score(f1_score_arr, precision_arr, recall_arr ,10)
print("macro_f1_score for train =  "+str(macro_f1_score_train)+'\n')
print("macro_precision for train =  "+str(macro_precision_train)+'\n')
print("macro_recall for train =  "+str(macro_recall_train)+'\n')
confusion_matrix_train=confusion_matrix(hot_form_y,hot_form_pred)
print("confusion matrix for train --->"+'\n'+str(confusion_matrix_train)+'\n')
visualise_confusion_for_mnist(confusion_matrix_train)
plt.show()

############ if mode =1 and n_epoch=0 comment the following lines######## 
# x = np.arange(n_epochs)
# plt.xlabel('epoches')
# plt.ylabel('accuracy')
# plt.plot(x, accuracy_list[0])
# plt.plot(x, accuracy_list[1])
# plt.legend(['training data', 'dev data'], loc='upper left')
# plt.show()

#####################################################################

test_accuracy = 0
out = net(X_test)
preds_test = np.argmax(out, axis=1).reshape(-1, 1)
test_accuracy = 100*(preds_test == y_test).sum() / 10000
print('test_accuracy = %d %%' % (test_accuracy))
# print((y_test==preds_test).all())


preds_test = np.argmax(out, axis=1).reshape(-1, 1)
micro_f1_test = micro_F1_SCORE(y_test,preds_test)
print("micro F1 score for test = micro precision = micro recall = "+str(micro_f1_test)+'\n')

hot_form_y_test=hot_form(y_test,10)
hot_form_pred_test=hot_form(preds_test,10)

f1_score_arr_test, precision_arr_test, recall_arr_test =f1_score_labels(hot_form_y_test ,hot_form_pred_test)
print("f1 score for test = "+str(f1_score_arr_test)+'\n')
print("precision for test = "+str(precision_arr_test)+'\n')
print("recall for test = "+str(recall_arr_test)+'\n')
macro_f1_score_test,macro_precision_test,macro_recall_test = macro_f1_score(f1_score_arr_test, precision_arr_test, recall_arr_test,10)
print("macro_f1_score for test=  "+str(macro_f1_score_test)+'\n')
print("macro_precision for test =  "+str(macro_precision_test)+'\n')
print("macro_recall for test =  "+str(macro_recall_test)+'\n')
confusion_matrix_test=confusion_matrix(hot_form_y_test,hot_form_pred_test)
print("confusion matrix for test --->"+'\n'+str(confusion_matrix_test)+'\n')
visualise_confusion_for_mnist(confusion_matrix_test)
plt.show()