import numpy as np
import matplotlib.pyplot as plt


def chi(y,s):
    def chientry(yy,ss):
        y1=(yy+1)/2
        y2=(1-yy)/2
        l1=y1*(yy-1)**2
        l2=y2*(yy+1)**2
        return (l1+l2)/ss**2
    return np.mean([chientry(a,b) for a,b in zip(y,s)])
def chinorm(y,s)->"s":
    """this sigmas require normalisation. One way to do this is by requiring that the chi**2 is one"""
    c=chi(y,s)
    s*=np.sqrt(c)
    return s

f=np.load("compare.npz")
xx=f["x"]
yy=f["y"]

bins=0#if bins is zero: this is the probability for one event. If bins is bigger than zero, sigma gets modified to more closely resemble the equivalent statistics, assuming this is generated in bins bins.
bins=len(yy)#if the number of bins is equal to the amount of comparison points (and you neglect their involvement), you can roughly compare there accuracy (68% should be in a sigma range)

f=np.load("nprob.npz")
x=f["x"]
y=f["y"]
if "s" in f.files:
    s=f["s"]
    s=s**1.5
    s=chinorm(y,s)
    print(chi(y,s))
    if bins>0:s/=np.sqrt(len(s)/bins)
else:
    s=None


if s is None:
    plt.plot(x,y,"o")
else:
    plt.errorbar(x,y,yerr=s,color="blue",alpha=0.5)
    plt.plot(x,y,"o",color="darkblue")
    plt.ylim([-1.1,1.1])


plt.plot(xx,yy,"o")
plt.show()




