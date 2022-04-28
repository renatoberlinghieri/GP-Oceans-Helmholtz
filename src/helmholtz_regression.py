
# Hide any GPUs to that tensorflow uses CPU (typically preferable due to memory constraints)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np

#first, define function for kernel
@tf.function
def Sqr_exp_2D(X1, Y1, X2, Y2, ls, sigma):
    """Sqr_exp_2D computes the covariances matrix for (X1, Y1) with (X2, Y2) under a squared exponential 
    kernel with a single length scale.
    
    Args:
        X1, Y1, X2, Y2 : np.arrays of shape [N, M]
        ls: length scale
        sigma: signal variance
    """
    d_sqr = (X1-X2)**2 + (Y1-Y2)**2
    K = sigma**2 * tf.exp(-(1./2.)*(d_sqr)/ls**2)
    return K

@tf.function
def K_mixed_partials(X1, Y1, X2, Y2, first_partial_X=True, second_partial_X=True, ls=1., sigma=1.):
    """K_mixed_partials returns mixed partial derivatives of the kernel function with respect to its arguments.
    
    The flags first_partial_X and second_partial_X determine if these derivatives are with respect to X or Y.
    With both set to true, we have
        second_partials[i,j] = \partial^2 / (\partial X1[i] * \partial X2[j] ) K((X1[i], Y1[i]), (X2[j], Y2[j]))
    """
    with tf.GradientTape() as t1:
        t1.watch(X2 if second_partial_X else Y2)
        with tf.GradientTape() as t2:
            t2.watch(X1 if first_partial_X else Y1)
            K = Sqr_exp_2D(X1, Y1, X2, Y2, ls=ls, sigma=sigma)
        first_partials = t2.gradient(K, X1) if first_partial_X else t2.gradient(K, Y1)
    second_partials = t1.gradient(first_partials, X2) if second_partial_X else t1.gradient(first_partials, Y2)
    return second_partials

@tf.function
def Sqr_exp_derivative_2D(XY, ls, sigma, obs_noise, curl=False, include_linear=False):
    """Sqr_exp_derivative_2D computes the covariance matrix of the gradients of a function
    distributed according of a GP with a squared exponential kernel, evaluated at XY.
    """
    print("N, i.e. XY.shape[0]: ", XY.shape[0])
    N = XY.shape[0] # number of locations to consider, e.g. grid_pts**2
    X, Y = (XY[:, :1], XY[:, 1:]) if not curl else (XY[:, 1:], XY[:, :1])
    X1, Y1 = tf.tile(X, [1, N]), tf.tile(Y, [1, N]) # each column is a copy of X / Y
    X2, Y2 = tf.transpose(X1), tf.transpose(Y1)
    
    Kxx = K_mixed_partials(X1, Y1, X2, Y2, first_partial_X=True, second_partial_X=True, ls=ls, sigma=sigma)
    Kxy = K_mixed_partials(X1, Y1, X2, Y2, first_partial_X=True, second_partial_X=False, ls=ls, sigma=sigma)
    Kyy = K_mixed_partials(X1, Y1, X2, Y2, first_partial_X=False, second_partial_X=False, ls=ls, sigma=sigma)
    
    if include_linear:
        c = 500
        Kxx += c
        Kyy += c
    
    K_all = tf.concat(
        [tf.concat([Kxx, Kxy], axis=1),
        tf.concat([tf.transpose(Kxy), Kyy], axis=1)], axis=0)
    
    if curl:
        # curl_vec looks like [1,1,1,..., 1, -1, ..., -1, -1, -1]
        curl_vec = tf.cast(tf.repeat(np.array([1., -1.]), [N, N]), tf.float32) #need to do this for dealing with tf
        K_all = curl_vec[None]*(curl_vec[:,None]*K_all)
    return K_all + tf.eye(K_all.shape[0]) * obs_noise**2

#set functions for GP regression
# Define kernel function for vector observations, operating on 2D coordinates
#old kernel_fcn = lambda XY: Sqr_exp_derivative_2D(XY, ls_Phi, sigma_Phi) + Sqr_exp_derivative_2D(XY, ls_A, sigma_A, curl=True)

def kernel_fcn(XY, ls_Phi = tf.constant(0.1, dtype=tf.float32), sigma_Phi = tf.constant(1., dtype=tf.float32),
              ls_A = tf.constant(0.1, dtype=tf.float32), sigma_A = tf.constant(1., dtype=tf.float32),
              obs_noise = tf.constant(0.02, dtype=tf.float32),
              include_linear = False):
    ker = Sqr_exp_derivative_2D(XY, ls_Phi, sigma_Phi, obs_noise, include_linear=include_linear) + Sqr_exp_derivative_2D(XY, ls_A, sigma_A, obs_noise, curl=True, include_linear=include_linear)
    return ker

def conditional(K_all, train_idcs, train_obs, test_idcs):
    """conditional returns the mean and covariance of the test observations 
    conditioned on the training observations.
    
    The mean is assumed to be zero.
    
    Args:
        K_all: covariance matrix for all datapoints.
        train_idcs, test_idcs: row / column indices of train and test datapoints.
        train_obs: values of training observations upon which to condition.
        
    Returns:
        Conditional expecation, variance of test observations, and likelihood.
    """
    print("K ALL SHAPE: ", K_all.shape)
    K_tr_tr = K_all[train_idcs][:, train_idcs] # train / train
    K_te_tr = K_all[test_idcs][:, train_idcs]  # test  / train
    K_te_te = K_all[test_idcs][:, test_idcs]   # test  / test
    print("K TR TR SHAPE: ", K_tr_tr.shape)
    print("K TE TR SHAPE: ", K_te_tr.shape)
    print("K TE TE SHAPE: ", K_te_te.shape)
    
    #cast to float64 for numerical issues with inverse
    K_te_tr = tf.cast(K_te_tr, dtype=tf.float64)
    K_tr_tr = tf.cast(K_tr_tr, dtype=tf.float64)
    K_te_te = tf.cast(K_te_te, dtype=tf.float64)
    train_obs = tf.cast(train_obs, dtype=tf.float64)
   
    #ipdb.set_trace()
    
    test_mu = tf.matmul(K_te_tr, tf.reshape(np.linalg.solve(K_tr_tr,train_obs),[-1,1]))
    test_cov = K_te_te - tf.matmul(K_te_tr, np.linalg.solve(K_tr_tr,tf.transpose(K_te_tr)))
    #ipdb.set_trace()
    log_likelihood = -0.5*tf.matmul(tf.transpose(tf.reshape(train_obs,[-1,1])),tf.reshape(np.linalg.solve(K_tr_tr,train_obs),[-1,1])) - 0.5*np.linalg.slogdet(K_tr_tr)[1]
    
    #cast back to float32 for consistency across notebook
    K_te_tr = tf.cast(K_te_tr, dtype=tf.float32)
    K_tr_tr = tf.cast(K_tr_tr, dtype=tf.float32)
    K_te_te = tf.cast(K_te_te, dtype=tf.float32)
    train_obs = tf.cast(train_obs, dtype=tf.float32)
    
    return test_mu, test_cov, log_likelihood

def vectorfield_GP_Regression(XY_train, UV_train, XY_test, 
                              ls_Phi = tf.constant(0.1, dtype=tf.float32), sigma_Phi = tf.constant(1., dtype=tf.float32),
                              ls_A = tf.constant(0.1, dtype=tf.float32), sigma_A = tf.constant(1., dtype=tf.float32), 
                              obs_noise = tf.constant(0.02, dtype=tf.float32), include_linear=False):
    """vectorfield_GP_Regression
    
    Assumes zero mean function. 
    
    Args:
        XY_train: np.array of shape [N_train, 2] with X & Y coordinates of observations
        UV_train: np.array of shape [N_train, 2] with U & V flow observations
        XY_test: np.array of shape [N_test, 2] with X & Y coordinates of test observations, usually on a grid
        kernel_fcn: kernel function that takes in an array of shape [N, 2] and returns a size 
            [2*N, 2*N] kernel matrix
        
    Returns:
        posterior mean of size [2*N_test] and covariance of size [2*N_test, 2*N_test] for test points.
    """
    np.random.seed(123)
    N_train, N_test = XY_train.shape[0], XY_test.shape[0]
    XY_train_test = np.concatenate([XY_train, XY_test], axis=0)
    assert XY_train_test.shape[0] == (N_train + N_test)
    train_obs = np.concatenate([UV_train[:, 0], UV_train[:, 1]])
    
    K_all = np.array(kernel_fcn(XY_train_test, ls_Phi, sigma_Phi, ls_A, sigma_A, obs_noise, include_linear))
    train_idcs = list(np.arange(N_train)) +list((N_train+N_test) + np.arange(N_train))
    test_idcs = [i for i in range(K_all.shape[0]) if i not in train_idcs]
    
    test_mu, test_cov, ll = conditional(K_all, train_idcs, train_obs, test_idcs)
    
    return test_mu, test_cov, ll



