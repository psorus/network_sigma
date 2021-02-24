import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

divergence_freeze=0.001#0+1.0
sigma_power=8.0
sigma_matter=0.025
loss_power=2.0#1.0



inputs = keras.Input(shape=(1,))
q=inputs

for zw in [4,12,30,80,30,12,4]:
    q=layers.Dense(zw,activation="relu")(q)

outputs=layers.Dense(2,activation="softmax")(q)
sigma=layers.Dense(1,activation="relu")(q)

outputs=keras.layers.Concatenate(axis=-1)([outputs,sigma])

model=keras.Model(inputs=inputs,outputs=outputs,name="simplefunction")

model.summary()

f=np.load("data.npz")
x=f["x"]
y=f["y"]

def ohot(q):
    if q<0:return [1.0,0.0]
    return [0.0,1.0]
y=[ohot(zw) for zw in y]

y=np.array(y)


def loss(y_true,y_pred,K=K):
    #print(dir(y_pred))
    #print(dir(K))
    #exit()
    a=tf.unstack(y_pred,axis=-1)
    #print(len(a),y_pred.shape)
    a1,a2,a3=a[0],a[1],a[2]
    val=tf.stack((a1,a2),axis=(1))
    sig=a3

    #remove unphysical divergences
    sig=K.abs(sig)

    #look at delta, since the softmax reduces the dimension anyway, and this way this is actually related to the sigma of the difference
    delta_val=val[:,0]-val[:,1]
    delta_tru=y_true[:,0]-y_true[:,1]

    loss=K.abs(delta_tru-delta_val)**loss_power
    sigma_loss=loss/(divergence_freeze+sig)
    sigma_loss=K.abs(sigma_loss-1)**sigma_power

    return K.mean(loss)+sigma_matter*sigma_loss



#test_inn=y[:100]
#test_out=np.concatenate((test_inn,np.random.normal(1.0,0.2,[100,1])),axis=-1)

#print(test_inn.shape)
#print(test_out.shape)

#print(loss(test_inn,test_out,np))


#exit()






model.compile(
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss=loss,
    optimizer=keras.optimizers.RMSprop(),
    metrics=[],
)

history = model.fit(x, y,
                    batch_size=50,
                    epochs=150,
                    validation_split=0.2,
                    callbacks=keras.callbacks.EarlyStopping(patience=3))

test_scores = model.evaluate(x, y, verbose=2)
print("Final loss:", test_scores)
#print("Test accuracy:", test_scores[1])



xv=np.arange(np.min(x),np.max(x),0.1)
yv=model.predict(xv)

sigma=yv[:,2]**(1/loss_power)#because less bad gradients this way
yv=yv[:,1]-yv[:,0]


np.savez_compressed("nprob",x=xv,y=yv,s=sigma)


print(xv.shape,yv.shape)













