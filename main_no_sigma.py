import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(1,))
q=inputs

for zw in [4,12,30,80,30,12,4,2]:
    q=layers.Dense(zw,activation="relu")(q)

q=layers.Dense(2,activation="softmax")(q)


model=keras.Model(inputs=inputs,outputs=q,name="simplefunction")

model.summary()

f=np.load("data.npz")
x=f["x"]
y=f["y"]

def ohot(q):
    if q<0:return [1.0,0.0]
    return [0.0,1.0]
y=[ohot(zw) for zw in y]

y=np.array(y)





model.compile(
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss="mse",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x, y, batch_size=50, epochs=5, validation_split=0.2)

test_scores = model.evaluate(x, y, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])



xv=np.arange(np.min(x),np.max(x),0.1)
yv=model.predict(xv)

yv=yv[:,1]-yv[:,0]


np.savez_compressed("nprob",x=xv,y=yv)


print(xv.shape,yv.shape)













