import matplotlib.pyplot as plt
import numpy as np

f=np.load("data.npz")
x=f["x"]
y=f["y"]

plt.plot(x,y,"o")

plt.show()


