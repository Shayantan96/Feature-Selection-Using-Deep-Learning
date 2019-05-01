import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import sklearn.model_selection as sk
import csv
from pickle import load

dim=500                      
n_x = 4096
n_y = 6
#m_total = 500

save_model = 0
regularize = 1
print_cost = 1
plot_func = 1
print_shape = 1

lambd = 0.01
num_epochs = 10
mini_batch_size = 64


def compute_cost(ZL, Y, regularizer):
    reg_term = 0
    cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(ZL), labels = tf.transpose(Y)))
    if regularize == 1:    
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term += tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    cost = cost1 + reg_term 
    return cost

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4'] 
    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)                                           
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              
    A2 = tf.nn.relu(Z2)                                           
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                              
    A3= tf.nn.relu(Z3)                                           
    Z4 = tf.add(tf.matmul(W4,A3),b4)
    return Z4

def random_mini_batches(X, Y):
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[ :, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[ :, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[ :, num_complete_minibatches*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[ :, num_complete_minibatches*mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001):
    ops.reset_default_graph()                         
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []                                        
    #create placeholders
    X = tf.placeholder(tf.float32, [n_x, None], name = "X")
    Y = tf.placeholder(tf.float32, [n_y, None], name = "Y")
    #initialize parameters
    regularizer = tf.contrib.layers.l2_regularizer(scale= lambd)    
    W1 = tf.get_variable("W1", [1000,n_x], regularizer = regularizer, initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [1000,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [1000,1000], regularizer = regularizer, initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [1000,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1000,1000], regularizer = regularizer, initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1000,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [n_y,1000], regularizer = regularizer, initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [n_y,1], initializer = tf.zeros_initializer())
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}    
    ZL = forward_propagation(X, parameters)
    cost = compute_cost(ZL, Y, regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / mini_batch_size) 
            minibatches = random_mini_batches(X_train, Y_train)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == 1 and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch+1, epoch_cost))
            if plot_func == 1 and epoch % 1 == 0:
                costs.append(epoch_cost)

        if plot_func == 1:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
        #parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train})*100, '%')
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test})*100, '%')
        if save_model == True:
            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            path = 'C:/Users/Sayantan/AnacondaProjects/Feature_Seelection'
            saver.save(sess, path, global_step = num_epochs)         
        return
    
def trim_xy(Ci, X):
    X=X.T
    arg = np.argsort(Ci)
    
    new_X = np.zeros((dim, X.shape[1]))
    for i in range(dim):
        new_X[i,:] = X[arg[i],:]
    print("Previous dimension of X:" + str(X.shape))    
    print("New dimension of X:" + str(new_X.shape))    
    return new_X.T

def read_csv(filename):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting = csv.QUOTE_NONE)
    y = np.array(list(reader))
    return y

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
def main():
    Ci = np.array(load(open('Ci.pkl','rb')))
    Ci=Ci.reshape((4096,))
    f1 = np.array(pd.read_csv("alexrelu1.csv"))
    f2 = np.array(pd.read_csv("alexrelu2.csv"))
    f3 = np.array(pd.read_csv("alexrelu3.csv"))
    f4 = np.array(pd.read_csv("alexrelu4.csv"))
    y = f4[:,-1]
    f4 = f4[:,:-1]
    x = np.concatenate((np.concatenate((np.concatenate((f1,f2), axis=1),f3), axis=1),f4), axis=1)
    onehot_encoder = OneHotEncoder(sparse=False)
    y = y.reshape(len(y), 1)
    y = onehot_encoder.fit_transform(y)
    
    x = trim_xy(Ci, x)
    
    
    X_train, X_test, Y_train, Y_test = sk.train_test_split(x, y, test_size = 0.2)   
    if print_shape == True:
        print("Shape of X_train is: " + str(X_train.shape))
        print("Shape of Y_train is: " + str(Y_train.shape))
        print("Shape of X_test is: " + str(X_test.shape))
        print("Shape of Y_test is: " + str(Y_test.shape))        
    model(X_train.T, Y_train.T, X_test.T, Y_test.T)
        
main()
print("done")