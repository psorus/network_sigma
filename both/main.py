import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from simple import Simple


divergence_freeze=0.001#0+1.0
scale_freeze=0.1
multiply_sigma=0.0+1.0
sigma_power=8.0
sigma_matter=0.025
loss_power=2.0#1.0
scale_power=4
scale_matter=0.05
scale_log=False
wid_matter=0.1*scale_matter#should not need to much importance, as keeping sigma const is more or less free
wid_power=2


inputs = keras.Input(shape=(1,))
q=inputs

for zw in [4,12,30,80,30,12,4]:
    q=layers.Dense(zw,activation="relu")(q)

outputs=layers.Dense(2,activation="softmax")(q)
sigma=layers.Dense(1,activation="relu")(q)
scale=layers.Dense(1,activation="relu")(q)
#scale=Simple(1)(q)#this would be a lot better, but sadly does not work for some reason

outputs=keras.layers.Concatenate(axis=-1)([outputs,sigma,scale])

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


def limitloss(q,K=K):#yeah I know...not actually limited, and kinda kills the scale power
    if scale_log:return K.log(q+1)
    return q


def loss(y_true,y_pred,K=K):
    #print(dir(y_pred))
    #print(dir(K))
    #exit()
    a=tf.unstack(y_pred,axis=-1)
    #print(len(a),y_pred.shape)
    a1,a2,a3,a4=a[0],a[1],a[2],a[3]

    val=tf.stack((a1,a2),axis=(1))
    sig=a3
    scal=a4

    #remove unphysical divergences#yes relu should do this already, but just to be save...
    sig=K.abs(sig)
    scal_var=K.abs(K.max(scal)-K.min(scal))**wid_power
    scal=K.mean(K.abs(scal))#also mean this, since for normalisation we only need one value (and the overall value is meaningless anyway, as test68 proved)


    #look at delta, since the softmax reduces the dimension anyway, and this way this is actually related to the sigma of the difference
    delta_val=val[:,0]-val[:,1]
    delta_tru=y_true[:,0]-y_true[:,1]




    loss=K.abs(delta_val-delta_tru)**loss_power#just mae for simpler math

    scale_loss=loss/(scale_freeze+scal)
    scale_loss=K.abs(scale_loss-1)**scale_power

    scale_loss=limitloss(scale_loss)





    loss/=(divergence_freeze+multiply_sigma*sig)#introduce sigma term, add constant to remove diverges


    #main loss term
    loss=K.mean(loss)#mean, to have less dependency on the batch size

    #assure that sigma is about 1
    loss+=sigma_matter*K.mean(K.abs(sig-1))**sigma_power#normalise sigma to be on average 1. low differences dont matter, but high ones do very much->high power

    #let scale be rigth
    loss+=scale_matter*scale_loss

    #let the scale be constant (simple would obviously not need this)
    loss+=wid_matter*scal_var

    return loss+K.mean(scal)


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

sigma=yv[:,2]
scale=yv[:,3]**(1/loss_power)

print("scale",scale)


yv=yv[:,1]-yv[:,0]


np.savez_compressed("nprob",x=xv,y=yv,s=sigma,scale=scale)


print(xv.shape,yv.shape)













