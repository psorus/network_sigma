import numpy as np
from numpy.random import randint
from tqdm import tqdm

md=100

def rnd():
    return (2*randint(0,2)-1)

def classify(x):
    if x<-md:return -1
    if x>md:return 1
    return 0

def mutal(x,dep=10):
    if dep<=0:return 0
    if (ac:=classify(x))!=0:return ac
    return (mutal(x-1,dep-1)+mutal(x+1,dep-1))/2

def mutatual(x):
    ac=x
    while (rel:=classify(ac))==0:
        ac+=rnd()
    return rel

def highstat(x):
    return np.mean([mutatual(x) for i in range(1000)])



x=np.arange(-md,1.0001*md,md/4)
#y=[mutal(zw) for zw in x]
yy=[]#[highstat(zw) for zw in x]

for zw in tqdm(x):
    yy.append(highstat(zw))


np.savez_compressed("compare",x=x,y=yy)



#print(y)
print(yy)




#class 1:below -1, class 2: above 1




