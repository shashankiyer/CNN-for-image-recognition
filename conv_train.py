from PIL import Image
import tensorflow as tf
import os
import math
import time, random
import sys, re
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
MINI_BATCH_SIZE = 5
EPOCHS = 5
LEARNING_RATE = 0.01

class conv_net:

    def __init__(self, letter, data_folder, model_file):
        """Creates a conv_net
         
        :param letter to train or test for
        :param train or test data
        :param tensorflow model file
        """
        if os.path.isdir(data_folder) is not True:
            print ("NO DATA")
            exit(1)

        self.letter = str(letter)        
        self.import_data(data_folder)
        self.model_file = os.path.join(os.getcwd(), model_file)

    def __image_reshape(temp_image):
        """Flattens the image into a 
           1D list

        :param The image to flatten
        :return 1D list
        """

        reshaped = np.array(temp_image.getdata()).reshape(temp_image.size[0], temp_image.size[1], 1)
        reshaped = reshaped.tolist()

        return reshaped
            

    def import_data(self, data_folder):
        """Extracts image data and stores it in a list

        :param The name of the directory containing the data
        """
        self.x=[]
        self.y=[]

        #Expression to identify positive samples
        positive_example = re.compile("[0-9]+_"+ self.letter +".png")

        print("Reading input data")

        data_folder = os.path.join( os.getcwd() , data_folder)
        for images in os.listdir(data_folder):
            if positive_example.match(images):
                self.y.append([0, 1])
            else:
                self.y.append([1, 0])
            images = os.path.join( data_folder , images)
            temp_image = Image.open(images)
            self.x.append(conv_net.__image_reshape(temp_image))

        #randomises data
        for i in range(1, len(self.x)-1):
            j = random.randint(i+1, len(self.x)-1)
            self.swap_t(i, j)
        
        print("Data extraction complete")

    def swap_t (self , i , j):
        '''Works as a swap function
           
        :param The ith and jth co-ordinate to swap
        '''
        tempx = self.x[i]
        self.x[i] = self.x[j]
        self.x[j] = tempx
        tempy = self.y[i]
        self.y[i] = self.y[j]
        self.y[j] = tempy
            

    def make_conv2d(x, W, b, kernel, features):
        """Creates a 2D conv and maxpool layer

        :param input to the conv layer
        :param weights, bias
        :param nxn kernel size
        :param number of features

        :return the created hidden layer
        """
        x = tf.nn.conv2d(x, W, strides=[1, kernel, kernel, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')        

    def create_weights_biases(index, kernel, prev_features, features, mode):
        """Creates weights and biases to be used in the hidden layers

        :param dimensions of the weight and bias vector
        
        :return created weights and bias vector
        """
        index_w = 'W'+str(index)
        index_b = 'b'+str(index)
        if mode == 'conv2d':
            W = tf.get_variable(index_w, [kernel, kernel, prev_features, features], dtype=tf.float32 ,initializer= tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(index_b, [features], dtype=tf.float32, initializer= tf.contrib.layers.xavier_initializer())
        else:
            W = tf.get_variable(index_w, [kernel*kernel*prev_features, features], dtype=tf.float32 ,initializer= tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(index_b, [features], dtype=tf.float32, initializer= tf.contrib.layers.xavier_initializer())
        return W, b         

    def dense_layer(x, W, b, activation):
        """Creates a fully connected dense layer

        :param input to the dense layer
        :param weights, bias
        :param activation function to apply

        :return the created dense layer
        """
        x = tf.add(tf.matmul(x, W), b)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif activation == 'relu':
            return tf.nn.relu(x)
        else:
            return x

    def __initialise_variables(network_description, mode, cost):
        """Creates a neural network based on the 
           feature_size and num_features in the 
           network_description file

        :param filename
        :param mode, either training or testing
        :param regularisation to apply

        :return created NN architecture and variables
        """
        #placeholder initialiser
        x = tf.placeholder( tf.float32 , [ None , 25, 25, 1] , name = "x")
        
        y = tf.placeholder( tf.float32 , [ None , 2 ] , name = "y")

        #Dictionary to store weights and biases
        W={}
        b={}        

        file = open(network_description + '.txt','r')
        variables = file.readlines()
        out_layer = x
        prev_features = 1

        for j in range (len(variables)-1):
            temp= variables[j].split()
            W[j], b[j] = conv_net.create_weights_biases(j, int(temp[0]), prev_features, int(temp[1]), 'conv2d')
            out_layer=conv_net.make_conv2d(out_layer, W[j], b[j], int(temp[0]), int(temp[1]))
            prev_features = int(temp[1])
        
        j=len(variables)-1
        
        reshape_val= out_layer.get_shape().as_list()[1]
        #Flatten max_pool
        out_layer = tf.reshape(out_layer, [-1, reshape_val* reshape_val* prev_features])

        #Fully connected output layer1
        temp= variables[j].split()
        W[j], b[j] = conv_net.create_weights_biases(j, reshape_val, prev_features, int(temp[0]), 'dense')
        out_layer= conv_net.dense_layer(out_layer, W[j], b[j], 'relu')
        prev_features = int(temp[0])

        j= j+ 1
        #Output layer with sigmoid activation
        W[j], b[j] = conv_net.create_weights_biases(j, 1, prev_features, 2, 'dense')
        out_layer= conv_net.dense_layer(out_layer, W[j], b[j], '')
        
        regularization_dic = {"cross": tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(0.0),  W.values()), 
                    "cross-l1": tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(0.01), W.values()),
                    "cross-l2":  tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01), W.values()),
                    "test": 0}    
        
        reg = regularization_dic[cost]

        #cost function
        cf = tf.nn.softmax_cross_entropy_with_logits(
            logits = out_layer , labels = y) + reg     

        out_layer= tf.nn.softmax(out_layer)
        
        #predicts if the output is equal to its expectation 
        correctness_of_prediction = tf.equal(
            tf.argmax(out_layer, 1), tf.argmax(y, 1))

        #accuracy of the NN
        accuracy = tf.reduce_mean(
            tf.cast(correctness_of_prediction, tf.float32), name='accuracy')

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE) 
        train = optimizer.minimize(cf)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        if mode == 'training':
            return x, y, sess, train, accuracy, out_layer
        elif mode == 'test':
            return x, y, out_layer, accuracy

    def __train(params, xdata, ydata, model):
        '''Trains the network with the provided data

        :param all essential training parameters
        :param input data and target values
        :param name of model file to save the created tf graph
        '''
       
        x=params['x']
        y=params['y']
        sess=params['sess']
        train=params['train']
        accuracy=params['accuracy']

        start_time=time.time()
    
        for j in range (EPOCHS):
            training_acc=0
            print("EPOCH NUMBER: ", j+1)
            for k in range(0, len(xdata), MINI_BATCH_SIZE):
                current_batch_x_train = xdata[k:k+MINI_BATCH_SIZE]
                current_batch_y_train = ydata[k:k+MINI_BATCH_SIZE]

                _= sess.run(train,
                        {x: current_batch_x_train, y: current_batch_y_train})           


        train_time=time.time() - start_time
        
        saver= tf.train.Saver()
        saver.save(sess , model)
        
        print("Total training time= ", train_time, "seconds")

    def test(self, model_file, xdata=None, ydata=None, cost=None, params=None, network_description=None):
        """Tests either the model stored in the file or the one currently
           being trained by the 5_fold_cross_validator

           :param: tf neural network
           :return: The model's accuracy, confusion matrix
        """
        if params is None:
            x, y, out_layer, acc=conv_net.__initialise_variables(network_description, 'test', cost)
            xdata=self.x
            ydata=self.y

        else :
            x=params['x']
            y=params['y']
            out_layer=params['out_layer']
            acc=params['accuracy']

        # Get Tensorflow model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess , model_file)        
        print ("Model restored!")
        
        start_time=time.time()

        total_accuracy = sess.run(acc, {x: xdata, y: ydata})

        #the softmax layer's output
        prediction = tf.argmax(out_layer, 1)
        actual = tf.argmax(ydata, 1)

        pred, act = sess.run([prediction, actual],  {x: xdata, y: ydata})
        #print("Prediction ",pred)
        #print("Actual",act)

        confusion_matrix = sess.run(tf.confusion_matrix(
                act, pred), {x: xdata, y: ydata})

        duration = time.time() - start_time

        if params is None:
            print ("Total number of items tested on: ", len(xdata))
            print ("Total Accuracy over testing data: ", total_accuracy)
            print("Testing time: ", duration, " seconds")
            print("Confusion Matrix:\n", confusion_matrix)

        else :
            return  total_accuracy, confusion_matrix          
    
    def _5_fold_cross_validation(self, network_description, cost):
        '''Performs 5-fold cross validation

        :param text file containing the network description
        :param regularisation to apply
        '''
        x, y, sess, train, accuracy, out_layer = conv_net.__initialise_variables(network_description, 'training', cost)
        params={}
        params['x']=x
        params['y']=y
        params['sess']=sess
        params['train']=train
        params['accuracy']=accuracy
        params['out_layer']=out_layer

        subset_size = len(self.x) // 5
        subsets_x = []
        subsets_y = []
        for i in range(0, len(self.x) , subset_size):
            subset = self.x[i:i+subset_size]
            subsets_x.append(subset)
            subset = self.y[i:i+subset_size]
            subsets_y.append(subset)

        for j in range(5):
            train_set_x = []
            train_set_y = []
            test_set_x = []
            test_set_y = []
            for i in range(5):
                if i != j:
                    train_set_x.extend(subsets_x[i])
                    train_set_y.extend(subsets_y[i])
                    
                else:
                    test_set_x=subsets_x[i]
                    test_set_y=subsets_y[i]                    
            conv_net.__train(params, train_set_x, train_set_y, self.model_file)
            acc, matrix= self.test(self.model_file, test_set_x, test_set_y, cost, params)
            print(acc)

            print(matrix)      
        
if __name__ == '__main__' :
    if len(sys.argv) != 8:
        print ("Please supply the correct arguments")
        raise SystemExit(1)

LEARNING_RATE = float(sys.argv[3])
EPOCHS = int(sys.argv[4])

trainer = conv_net ( sys.argv[5],sys.argv[7],sys.argv[6])

if str(sys.argv[1]) == 'test' :
    trainer.test(model_file = str(sys.argv[6]), cost=sys.argv[1], network_description=str(sys.argv[2]))
else:
    trainer._5_fold_cross_validation(str(sys.argv[2]), sys.argv[1])
        
#python conv_train.py cost network_description epsilon max_updates class_letter model_file_name train_folder_name