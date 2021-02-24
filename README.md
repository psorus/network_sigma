# network_sigma
Originally just me playing around with using neuronal networks to find a very simple function (random walk probability to go to first to point 100, than to point -100 as function of the starting position), but evolving into an idea of generating sigmas for the output of a neuronal network. This works, by letting the network decide itself what it more certain of, and what it is not (and than using some clever math, aswell as a normalisation)


This is based on the fact, that sum((sigma_i/alpha_i)^2) where sum(alpha_i)=const is minimized at alpha_i\~sigma_i^(2/3). This means, by introducing a learnable division variable into the loss (that the network can decide to give any dynamic value to, see main.py), we can get the sigma of this variable by just applying a power 1.5. Sadly this still needs some kind of normalisation (which makes sense, since the standart deviation of a continuos variable is not so easily defined). We use here (nplot.py) the chi² test to assert that chi²~=1, but the standart deviation for a single point should work well to.

This is restricted to L2 losses (mse), but could be easily extended for LN losses (l1 for example follows alpha_1~sigma_i^(1/2)).



