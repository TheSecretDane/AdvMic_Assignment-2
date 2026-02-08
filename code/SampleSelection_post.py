import numpy as np
from numpy import linalg as la
from scipy.stats import norm
      
def q(theta,y,x): # theta indicates that we also pass in \eta parameter
    return -loglik_probit(theta,y,x)

def loglik_probit(theta, y, x): 
    #assert y.ndim == 1, f'y should be 1-dimensional'
    #assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    eta = theta[-1] # take abs() to ensure positivity ???
    
    #print(f'dimensions: eta.shape {eta.shape}')
    #print(eta)
    
    #hjaltes_x1 = x[:,1]
    #print(hjaltes_x1)
    # This is the term for when we include constant. I don't know if we should include constant 😅

    var_term = np.exp(x * eta)

    #print(f'dimensions: x.shape {x.shape}\n b.shape {b.shape}\nvar_term.shape {var_term.shape}')

    z = (x * b)
    z = z / var_term

    G = norm.cdf(z)

    # Make sure that no values are below 0 or above 1.
    h = np.sqrt(np.finfo(float).eps)
    G = np.clip(G, h, 1 - h)

    # Make sure g and y is 1-D array
    G = G.reshape(-1, )
    y = y.reshape(-1, )

    ll = y*np.log(G) + (1 - y)*np.log(1 - G)
    return ll

def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N,K = x.shape 
    b_ols = np.linalg.solve(x.T@x, x.T@y)
    res = y - x@b_ols 
    sig2hat = 1./(N-K) * np.dot(res, res)
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

"""
def loglikelihood(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity 
    N,K = x.shape
"""