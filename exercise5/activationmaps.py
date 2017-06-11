# deeplearninig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dlipr
import keras
from keras.layers import *
from keras import backend as K
from keras.utils.np_utils import to_categorical
import os





data = dlipr.ising.load_data()

# plot some examples
data.plot_examples(5, fname='examples.png')

# features: images of spin configurations
x_train = data.train_images
x_test = data.test_images

X_train=np.expand_dims(x_train, axis=-1)
X_test=np.expand_dims(x_test, axis=-1)


print X_train.shape
print X_train.shape[1:]
# classes: simulated temperatures
T = data.classes

# labels: class index of simulated temperature
# create binary training labels: T > Tc?
#temperatures= np.linspace(0,5,21)
#temperatures=np.append(temperatures, 2.27)
#print temperatures
testaccuracy=np.array([])

Tc=2.27

y_train = T[data.train_labels] > Tc
y_test = T[data.test_labels] > Tc
    
    
    
Y_train = to_categorical(y_train, 2)
Y_test = to_categorical(y_test,2)

model = keras.models.load_model('modelconv2_27.h5')
print(model.summary())



conv1 = model.layers[0]
act1  = model.layers[1]
conv2 = model.layers[4]
act2 = model.layers[5]
avepool=model.layers[6]

W1, b1 = conv1.get_weights()
W2, b2 = conv2.get_weights()


im=[1,2,3,5]

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap


#Use base cmap to create transparent
mycmap = transparent_cmap(plt.cm.Reds)
for i in im:   
    print i
    folder = 'task2/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    inputs = [K.learning_phase()] + model.inputs
    conv1_func = K.function(inputs, [conv1.output])
    conv2_func = K.function(inputs, [conv2.output])
    
    class_weights = conv2.get_weights()[0]
    #print class_weights
    #print "b2, W2"
    #print b2, W2
    # plot the activations for test sample i
    Xin = X_test[i][np.newaxis]
    Xout1 = conv1_func([0] + [Xin])[0][0]
    Xout2 = conv2_func([0] + [Xin])[0][0]
    
    cam = np.zeros(dtype = np.float32, shape = Xout2.shape[1:3])
    target_class = 1
    for j, w in enumerate(class_weights[:, target_class]):
        cam += w * Xout2[j, :, :]
    print cam.shape  
    cam /= np.max(cam)
    grid = np.resize(cam, (32, 32))
    #plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
    #           interpolation='nearest', cmap=cm.gist_rainbow)
    
    
    
    print x_test[i]
    print x_test[i].shape
    
    Yp = model.predict(X_test)
    yp = np.argmax(Yp, axis=1)
    
    dlipr.utils.plot_prediction(
        Yp[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname=folder+'image%i.png' % i)
    
    w, h = 32, 32
    y, x = np.mgrid[0:h, 0:w]
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(x_test[i])
    
    cb = ax.contourf(x, y, grid, 10, cmap=mycmap)
    plt.colorbar(cb)
    plt.savefig(folder+"map"+str(i)+".png")
