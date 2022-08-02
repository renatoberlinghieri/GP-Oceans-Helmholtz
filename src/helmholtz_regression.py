# Hide any GPUs to that tensorflow uses CPU (typically preferable due to memory constraints)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import time


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


def Sqr_exp_2D_uv(XY, ls, sigma, obs_noise):
    """
    Function similar to sqr_exp_2D, with different input structure. 
    To be used for the analysis without the Helmholtz decomposition.
    """
    N = XY.shape[0] # number of locations to consider, e.g. grid_pts**2
    
    X, Y = (XY[:, :1], XY[:, 1:])
    X1, Y1 = tf.tile(X, [1, N]), tf.tile(Y, [1, N]) # each column is a copy of X / Y
    X2, Y2 = tf.transpose(X1), tf.transpose(Y1)
    
    d_sqr = (X1-X2)**2 + (Y1-Y2)**2
    K = sigma**2 * tf.exp(-(1./2.)*(d_sqr)/ls**2)
    
    return K + tf.eye(K.shape[0]) * obs_noise**2


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

def kernel_fcn_helm(XY, ls_Phi = tf.constant(0.1, dtype=tf.float32), sigma_Phi = tf.constant(1., dtype=tf.float32),
              ls_A = tf.constant(0.1, dtype=tf.float32), sigma_A = tf.constant(1., dtype=tf.float32),
              obs_noise = tf.constant(0.02, dtype=tf.float32),
              include_linear = False):
    ker = Sqr_exp_derivative_2D(XY, ls_Phi, sigma_Phi, obs_noise, include_linear=include_linear) + Sqr_exp_derivative_2D(XY, ls_A, sigma_A, obs_noise, curl=True, include_linear=include_linear)
    return ker

def kernel_fcn_helm_mult_ls(XY, ls_Phi1 = tf.constant(0.1, dtype=tf.float32), 
                            sigma_Phi1 = tf.constant(1., dtype=tf.float32),
                            ls_A1 = tf.constant(0.1, dtype=tf.float32), 
                            sigma_A1 = tf.constant(1., dtype=tf.float32), 
                            ls_Phi2 = tf.constant(0.1, dtype=tf.float32), 
                            sigma_Phi2 = tf.constant(1., dtype=tf.float32),
                            ls_A2 = tf.constant(0.1, dtype=tf.float32), 
                            sigma_A2 = tf.constant(1., dtype=tf.float32),
                            obs_noise = tf.constant(0.02, dtype=tf.float32),
                            include_linear = False):
    ker = Sqr_exp_derivative_2D(XY, ls_Phi1, sigma_Phi1, obs_noise, include_linear=include_linear) + Sqr_exp_derivative_2D(XY, ls_A1, sigma_A1, obs_noise, curl=True, include_linear=include_linear) + Sqr_exp_derivative_2D(XY, ls_Phi2, sigma_Phi2, obs_noise, include_linear=include_linear) + Sqr_exp_derivative_2D(XY, ls_A2, sigma_A2, obs_noise, curl=True, include_linear=include_linear)
    return ker


def kernel_fcn_uv(XY, ls_u = tf.constant(0.1, dtype=tf.float32), sigma_u = tf.constant(1., dtype=tf.float32),
                ls_v = tf.constant(0.1, dtype=tf.float32), sigma_v = tf.constant(1., dtype=tf.float32),
                obs_noise = tf.constant(0.02, dtype=tf.float32)):
    """
    UV_GP_kernel computes a covariance kernel for a vector-field evaluated at points XY.

    Args:
    XY: points R^2 at which to compute covariance (np.array of shape [N, 2])
    ls_u, ls_v: length-scales of the covariances over U and V, respectively (scalars)
    sigma_u, sigma_v: signal standard deviations for U and V, respectively (scalars)
    sigma_obs: noise standard deviation for observations

    Returns:
        Covariance matrix K (np.array of shape [2*N, 2*N]) where for n in [N],
            K[n,m]=Cov(U[n], U[m]),
            K[N+n,N+m]=Cov(V[n], V[m]), and
            K(n, N+m)=K(N+n, m)=0.
    """
    N = XY.shape[0]
    
    K_all = tf.concat(
            [tf.concat([Sqr_exp_2D_uv(XY, ls_u, sigma_u, obs_noise), np.zeros([N,N])], axis=1),
            tf.concat([np.zeros([N, N]), Sqr_exp_2D_uv(XY, ls_v, sigma_v, obs_noise)], axis=1)], axis=0)
    return K_all

def conditional(K_all, train_idcs, train_obs, test_idcs, grid_search=False):
    """conditional returns the mean and covariance of the test observations 
    conditioned on the training observations.
    
    The mean is assumed to be zero.
    
    Args:
        K_all: covariance matrix for all datapoints.
        train_idcs, test_idcs: row / column indices of train and test datapoints.
        train_obs: values of training observations upon which to condition.
        
    Returns:
        Conditional expectation, variance of test observations, and likelihood.
    """
    K_tr_tr = K_all[train_idcs][:, train_idcs] # train / train
    K_te_tr = K_all[test_idcs][:, train_idcs]  # test  / train
    K_te_te = K_all[test_idcs][:, test_idcs]   # test  / test
    
    #cast to float64 for numerical issues with inverse
    K_te_tr = tf.cast(K_te_tr, dtype=tf.float64)
    K_tr_tr = tf.cast(K_tr_tr, dtype=tf.float64)
    K_te_te = tf.cast(K_te_te, dtype=tf.float64)
    train_obs = tf.cast(train_obs, dtype=tf.float64)
    
    #condition number for K_tr_tr
    #print(condition_number(K_tr_tr))
    
    log_likelihood = -0.5*tf.matmul(tf.transpose(tf.reshape(train_obs,[-1,1])),tf.reshape(np.linalg.solve(K_tr_tr,train_obs),[-1,1])) - 0.5*np.linalg.slogdet(K_tr_tr)[1]
    
    #to speed up grid_search, no need to compute the actual mean and cov:
    if grid_search:
        return 0, 0, log_likelihood
    
    #tstart = time.time()
    
    #ipdb.set_trace()
    test_mu = tf.matmul(K_te_tr, tf.reshape(np.linalg.solve(K_tr_tr,train_obs),[-1,1]))
    test_cov = K_te_te - tf.matmul(K_te_tr, np.linalg.solve(K_tr_tr,tf.transpose(K_te_tr)))
    
    #cast back to float32 for consistency across notebook
    K_te_tr = tf.cast(K_te_tr, dtype=tf.float32)
    K_tr_tr = tf.cast(K_tr_tr, dtype=tf.float32)
    K_te_te = tf.cast(K_te_te, dtype=tf.float32)
    train_obs = tf.cast(train_obs, dtype=tf.float32)
    
    #tend = time.time()
    #print("time gained inside conditional: ", tend-tstart)
    
    return test_mu, test_cov, log_likelihood

def condition_number(A):
    """
    Args:
        A: positive semidefinite matrix
    """
    eigval = np.linalg.eigvalsh(A)
    cn = np.log(np.max(eigval))-np.log(np.min(eigval))
    return cn 
    

def vectorfield_GP_Regression_helm(XY_train, UV_train, XY_test, 
                              ls_Phi = tf.constant(0.1, dtype=tf.float32), sigma_Phi = tf.constant(1., dtype=tf.float32),
                              ls_A = tf.constant(0.1, dtype=tf.float32), sigma_A = tf.constant(1., dtype=tf.float32), 
                              obs_noise = tf.constant(0.02, dtype=tf.float32), include_linear=False, grid_search=False):
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
    #tstart = time.time()
    np.random.seed(123)
    N_train, N_test = XY_train.shape[0], XY_test.shape[0]
    XY_train_test = np.concatenate([XY_train, XY_test], axis=0)
    assert XY_train_test.shape[0] == (N_train + N_test)
    train_obs = np.concatenate([UV_train[:, 0], UV_train[:, 1]])
    
    #tbeforekernel = time.time()
    K_all = np.array(kernel_fcn_helm(XY_train_test, ls_Phi, sigma_Phi, ls_A, sigma_A, obs_noise, include_linear))
    #tafterkernel = time.time()
    
    train_idcs = list(np.arange(N_train)) +list((N_train+N_test) + np.arange(N_train))
    test_idcs = [i for i in range(K_all.shape[0]) if i not in train_idcs]
    
    test_mu, test_cov, ll = conditional(K_all, train_idcs, train_obs, test_idcs, grid_search=grid_search)
    
    #tend = time.time()
    
    #print(f"overall time is: {tend-tstart}, where {tafterkernel - tbeforekernel} is used for kernel")
    return test_mu, test_cov, ll


def vectorfield_GP_Regression_helm_mult_ls(XY_train, UV_train, XY_test, 
                              ls_Phi1 = tf.constant(0.1, dtype=tf.float32), sigma_Phi1 = tf.constant(1., dtype=tf.float32),
                              ls_A1 = tf.constant(0.1, dtype=tf.float32), sigma_A1 = tf.constant(1., dtype=tf.float32), 
                              ls_Phi2 = tf.constant(10, dtype=tf.float32), sigma_Phi2 = tf.constant(1., dtype=tf.float32),
                              ls_A2 = tf.constant(10, dtype=tf.float32), sigma_A2 = tf.constant(1., dtype=tf.float32), 
                              obs_noise = tf.constant(0.02, dtype=tf.float32), include_linear=False, grid_search=False):
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
    #tstart = time.time()
    np.random.seed(123)
    N_train, N_test = XY_train.shape[0], XY_test.shape[0]
    XY_train_test = np.concatenate([XY_train, XY_test], axis=0)
    assert XY_train_test.shape[0] == (N_train + N_test)
    train_obs = np.concatenate([UV_train[:, 0], UV_train[:, 1]])
    
    #tbeforekernel = time.time()
    K_all = np.array(kernel_fcn_helm_mult_ls(XY_train_test, ls_Phi1, sigma_Phi1, ls_A1, sigma_A1, ls_Phi2, sigma_Phi2, ls_A2, sigma_A2, obs_noise, include_linear))
    #tafterkernel = time.time()
    
    train_idcs = list(np.arange(N_train)) +list((N_train+N_test) + np.arange(N_train))
    test_idcs = [i for i in range(K_all.shape[0]) if i not in train_idcs]
    
    test_mu, test_cov, ll = conditional(K_all, train_idcs, train_obs, test_idcs, grid_search=grid_search)
    
    #tend = time.time()
    
    #print(f"overall time is: {tend-tstart}, where {tafterkernel - tbeforekernel} is used for kernel")
    return test_mu, test_cov, ll


def vectorfield_GP_Regression_uv(XY_train, UV_train, XY_test, 
                                 ls_u = tf.constant(0.1, dtype=tf.float32), sigma_u = tf.constant(1., dtype=tf.float32),
                                 ls_v = tf.constant(0.1, dtype=tf.float32), sigma_v = tf.constant(1., dtype=tf.float32),
                                 obs_noise = tf.constant(0.02, dtype=tf.float32)):
    
    N_train, N_test = XY_train.shape[0], XY_test.shape[0]
    XY_train_test = np.concatenate([XY_train, XY_test], axis=0)
    assert XY_train_test.shape[0] == (N_train + N_test)
    train_obs = np.concatenate([UV_train[:, 0], UV_train[:, 1]])
    
    K_all = np.array(kernel_fcn_uv(XY_train_test, ls_u, sigma_u, ls_v, sigma_v, obs_noise))
    train_idcs = list(np.arange(N_train)) +list((N_train+N_test) + np.arange(N_train))
    test_idcs = [i for i in range(K_all.shape[0]) if i not in train_idcs]
    
    test_mu, test_cov, ll = conditional(K_all, train_idcs, train_obs, test_idcs)
    
    return test_mu, test_cov, ll


def plot_results_grid(X_grid, Y_grid, XY_train, test_mu, test_cov, best_params, grid_points, min_lat, min_lon, scale=0.03, helm=True):
    
    # format mean and variance for plotting
    test_var = np.diag(test_cov)
    test_mu_grid = tf.reshape(test_mu, [2, grid_points, grid_points])
    test_var = tf.reshape(test_var, [2, grid_points, grid_points])
    test_var_grid = test_var[0] + test_var[1]

    # Plot the predictive mean and variance conditioned on training points 
    f, axarr = plt.subplots(ncols=2, figsize=[11,5])
    ax = axarr[0]
    if helm:
        ax.set_title("Prediction ~ our method", fontsize=20)
    else:
        ax.set_title("Prediction ~ standard GP", fontsize=20)
    ax.quiver(X_grid, Y_grid, test_mu_grid[0], test_mu_grid[1], scale_units='xy', scale=scale)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.scatter(X_grid, Y_grid, label="predicted currents", c='black', s=0.001)
    #ax.quiver(XY_train[:, 0], XY_train[:, 1], UV_train[:,0], UV_train[:,1], color='red', scale_units='xy', scale=0.05)
    
    #to plot scale segments
    ax.plot(np.arange(min_lat+10, min_lat+10+best_params[0]), np.ones(best_params[0])*(min_lon+10), c = 'blue')
    ax.plot(np.ones(5)*(min_lat+10), np.arange(min_lon+8, min_lon+13), c = 'blue')
    ax.plot(np.ones(5)*(min_lat+10+best_params[0]), np.arange(min_lon+8, min_lon+13), c = 'blue')
    if helm:
        str_par1 = "Lengthscale Φ: " + str(best_params[0])
    else:
        str_par1 = "Lengthscale U: " + str(best_params[0])
    ax.annotate(str_par1, (min_lat+13+best_params[0], min_lon+7), size = 7, c = 'blue')
    
    ax.plot(np.arange(min_lat+10, min_lat+10+best_params[2]), np.ones(best_params[2])*(min_lon+3), c = 'green')
    ax.plot(np.ones(5)*(min_lat+10), np.arange(min_lon+1, min_lon+6), c = 'green')
    ax.plot(np.ones(5)*(min_lat+10+best_params[2]), np.arange(min_lon+1, min_lon+6), c = 'green')
    if helm:
        str_par2 = "Lengthscale Ψ: " + str(best_params[2])
    else:
        str_par2 = "Lengthscale V: " + str(best_params[2])
    ax.annotate(str_par2, (min_lat+13+best_params[2], min_lon), size = 7, c = 'green')
    
    ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    ax.legend()

    ax = axarr[1]
    ax.set_title("Uncertainty", fontsize=20)
    cs = ax.contourf(X_grid, Y_grid, test_var_grid)
    ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c='r', s=2)
    ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
    f.colorbar(cs, ax = ax, cmap='inferno')

    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return


def plot_posterior_draws(X_grid, Y_grid, XY_train, test_mu, test_cov, grid_points, N_samples=4):
    f, axarr = plt.subplots(ncols=N_samples, figsize=[5*N_samples, 5])
    for i, ax in enumerate(axarr):
        test_UV = np.random.multivariate_normal(mean=test_mu[:,0], cov=test_cov)
        test_UV_grid = test_UV.reshape([2, grid_points, grid_points])
        ax.quiver(X_grid, Y_grid, test_UV_grid[0], test_UV_grid[1])
        ax.scatter(XY_train[:, 0], XY_train[:, 1], label="buoys' locations", c = 'red', s = 1)
        ax.set_xlabel('Latitude'); ax.set_ylabel('Longitude')
        ax.set_title("Sample %d"%i)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return
