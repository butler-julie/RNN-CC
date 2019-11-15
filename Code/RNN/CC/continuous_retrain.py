from RNNSupport import *
import numpy as np
import matplotlib.pyplot as plt

# Vary Dimension
#datatype='VaryDimension'
#X_tot = np.arange(2, 42, 2)
#y_tot = np.array([-0.03077640549, -0.08336233266, -0.1446729567, -0.2116753732, -0.2830637392, -0.3581341341, -0.436462435, -0.5177783846,
#	-0.6019067271, -0.6887363571, -0.7782028952, -0.8702784034, -0.9649652536, -1.062292565, -1.16231451, 
#	-1.265109911, -1.370782966, -1.479465113, -1.591317992, -1.70653767])

# Vary Interaction Negative
#datatype='VaryDimensionNegative'
X_tot = np.arange(-1, 0, 0.05)
y_tot = np.array([-1.019822621,-0.9373428759,-0.8571531335,-0.7793624503,-0.7040887974,
    -0.6314601306,-0.561615627,-0.4947071038,-0.4309007163,-0.3703789126,-0.3133427645,
    -0.2600147228,-0.2106419338,-0.1655002064,-0.1248988336,-0.08918647296,-0.05875839719,
    -0.03406548992,-0.01562553455,-0.004037522178])

## mae_error/3 * i**2

# Vary Interaction Positive
#X_tot = np.arange(0.05, 0.85, 0.05)
#y_tot = np.array([-0.004334904077,-0.01801896484,-0.04222576507,-0.07838310563,-0.128252924,
#    -0.1940453966,-0.2785866456,-0.3855739487,-0.5199809785,-0.6887363571,-0.9019400869,-1.175251697,
#    -1.535217909,-2.033720441,-2.80365727,-4.719209688])


assert len(X_tot) == len(y_tot)




dim=12
total_points = 20
y_train = y_tot[:dim]
y_return = y_train.tolist()

seq = 2

hidden_neurons = 500
loss = 'mse'
optimizer = 'adam'
activation = 'relu'

i = 0

while len(y_return) < total_points:
    print ('CURRENT LENGTH: ', len(y_return))
    if len(y_return)%seq == 0:    
        model = dnn2_rnn3(seq, hidden_neurons, loss, optimizer, activation, 0.0)
        X_train, y_train = format_data (y_return, seq)
        iterations = 500

        model.fit (X_train, y_train, epochs=iterations, validation_split=0.0, verbose=False)


    next_input = np.array([[[y_return[-2]], [y_return[-1]]]])
    next_output = model.predict(next_input)
    y_return.append(next_output[0][0])

    i = i + 1


    if len(y_return) == 13:
        y_onetrain = []
        y_onetrain = y_tot[:dim].tolist()
        next_input = np.array([[[y_onetrain[-2]], [y_onetrain[-1]]]])
        last = [y_onetrain[-1]]

        while len(y_onetrain) < total_points:
            next = model.predict(next_input)
            y_onetrain.append(next[0][0])
            next_input = np.array([[last, next[0]]])
            last = next[0]

print('MSE With Retraining: ', mse(y_tot, y_return))
print('MSE Without Retraining: ', mse(y_tot, y_onetrain))

plt.plot(y_tot, 'r', linewidth=4)
plt.plot(y_return, 'b', linewidth=4)
plt.plot(y_onetrain, 'g', linewidth=4)
plt.show()








