# binomial probability mass function

from scipy.misc import comb
def pmf(n,k,p):
    return comb(n,k) * p**k * (1-p)**(n-k)
