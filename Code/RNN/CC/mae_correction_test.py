from RNNSupport import MAE_correction
from RNNSupport import MAE_correction4
from RNNSupport import  mae, mae_per_element, mse
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


dim = 12
y_data = y_tot[:dim]

y = []


#for i in range (0, 20):
#    y_pred = MAE_correction (y_data, i, 2, 200, 'mse', 'adam', 500, 20)
#    err = mae(y_tot, y_pred)
#print(err)
#    y.append(err)
#    print ('I, MAE: ', i, err)

y_pred = MAE_correction (y_data, 1, 2, 200, 'mse', 'adam', 500, 20)
y_pred4 = MAE_correction4 (y_data, 2, 2, 200, 'mse', 'adam', 500, 20)







y = mae_per_element(y_tot, y_pred)
print(y)



plt.plot(y_pred[dim:], 'r', linewidth=4, label='Corrections')
plt.plot(y_pred4[dim:], 'y', linewidth=4, label='No Correction')
plt.plot(y_tot[dim:], 'm', linewidth=4, label='Actual')

plt.legend()

plt.show()
