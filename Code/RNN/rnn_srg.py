# For matrices and calculations
import numpy as np
# For machine learning (backend for keras)
import tensorflow as tf
# User-friendly machine learning library
# Front end for TensorFlow
import keras
# Different methods from Keras needed to create an RNN
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation 

from keras.layers.recurrent import SimpleRNN
from timeit import default_timer as timer

X_tot = np.arange(0, 10.00, 0.001)
#print(X_tot)
data = np.load('H.npy').tolist()[0:10000]
y_tot = [i[0] for i in data]


assert len(X_tot) == len(y_tot)

dim=100

X_train = X_tot[:dim]
y_train = y_tot[:dim]
#print(y_train)

# FORMAT_DATA
def format_data(data, length_of_sequence = 2):  
    """
        Inputs:
            data(a numpy array): the data that will be the inputs to the recurrent neural
                network
            length_of_sequence (an int): the number of elements in one iteration of the
                sequence patter.  For a function approximator use length_of_sequence = 2.
        Returns:
            rnn_input (a 3D numpy array): the input data for the recurrent neural network.  Its
                dimensions are length of data - length of sequence, length of sequence, 
                dimnsion of data
            rnn_output (a numpy array): the training data for the neural network
        Formats data to be used in a recurrent neural network.
    """

    X, Y = [], []
    for i in range(len(data)-length_of_sequence):
        # Get the next length_of_sequence elements
        a = data[i:i+length_of_sequence]
        # Get the element that immediately follows that
        b = data[i+length_of_sequence]
        # Reshape so that each data point is contained in its own array
        a = np.reshape (a, (len(a), 1))
        X.append(a)
        Y.append(b)
    rnn_input = np.array(X)
    rnn_output = np.array(Y)

    return rnn_input, rnn_output

# Generate the training data for the RNN
rnn_input, rnn_training = format_data(y_train, 2)
#print(rnn_input)


def rnn(length_of_sequences, batch_size = None, stateful = False):
    in_out_neurons = 1
    hidden_neurons = 200
    inp = Input(batch_shape=(batch_size, 
                length_of_sequences, 
                in_out_neurons))  

    rnn = SimpleRNN(hidden_neurons, 
                    return_sequences=False,
                    stateful = stateful,
                    name="RNN")(inp)

    dens = Dense(in_out_neurons,name="dense")(rnn)
    model = Model(inputs=[inp],outputs=[dens])
    
    model.compile(loss="mean_squared_error", optimizer="adam")

    
    return(model,(inp,rnn,dens))
## use the default values for batch_size, stateful
model, (inp,rnn,dens) = rnn(length_of_sequences = rnn_input.shape[1])
#model.summary()

start = timer()
hist = model.fit(rnn_input, rnn_training, batch_size=None, epochs=250,
                 verbose=False,validation_split=0.05)

import matplotlib.pyplot as plt

#for label in ["loss","val_loss"]:
#    plt.plot(hist.history[label],label=label)

#plt.ylabel("loss")
#plt.xlabel("epoch")
#plt.title("The final validation loss: {}".format(hist.history["val_loss"][-1]))
#plt.legend()
#plt.show()

def test_rnn (x1, y_test, plot_min, plot_max):
    #y_pred = [y_test[0], y_test[1]]
    #current_pred = np.array([[[y_test[0]], [y_test[1]]]])
    #last = np.array([[y_test[1]]])
    #for i in range (2, len(y_test)):
    #    next = model.predict(current_pred)
    #    y_pred.append(next)
    #    current_pred = np.asarray([[last.flatten(), next.flatten()]])
    #    last = next

    #assert len(y_pred) == len(y_test)
        
    #plt.figure(figsize=(19,3))
    #plt.plot([10, 10, 10], [1.5, 0, -1.5])
    X_test, a = format_data (y_test, 2)
    #print X_test[]
    y_pred = model.predict(X_test)

    x1 = x1[2:]
    y_test = y_test[2:]

    #y_pred = y_test[:dim]
    #next_input = np.array([[[y_test[dim-2]], [y_test[dim-1]]]])
    #print(next_input)
    #last = [y_test[dim-1]]

    #for i in range (dim, len(y_test)):
        #print 'ITER: ', i
    #    next = model.predict(next_input)
    #    y_pred.append(next[0][0])
        #print(next)
        #print 'DIFF: ', next[0][0]-y_test[i]
    #    next_input = np.array([[last, next[0]]])
    #    last = next

    print('MSE: ', np.square(np.subtract(y_test, y_pred.flatten())).mean())
#    name = 'Predicted'+str(dim)+'.csv'
#    #np.savetxt(name, y_pred, delimiter=',')
    fig, ax = plt.subplots()
    ax.plot(x1, y_test, label="true", linewidth=2)
    ax.plot(x1, y_pred, 'g-.',label="predicted", linewidth=4)
    ax.legend()

    ax.axvspan(plot_min, plot_max, alpha=0.25, color='red')
    plt.show()

    print ('Actual: ', y_test[2000])
    print('Predicted: ', y_pred[2000][0])
    
    #diff = y_test - y_pred.flatten()

    #plt.plot(x1, diff, linewidth=4)
    #plt.show()

test_rnn(X_tot, y_tot, X_tot[0], X_tot[dim-1])
end = timer()
print('Time: ', end-start)

