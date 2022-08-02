import helmholtz_regression as hr
import tensorflow as tf
import numpy as np
from sklearn.model_selection import ParameterGrid

def grid_search_helm(ls_Phi_grid, sigma_Phi_grid, ls_A_grid, sigma_A_grid, obs_noise_grid, 
                XY_train, UV_train, XY_test, file_name="marginal_ll_gs_helm"):
    """
    function to perform grid search of parameters for GP regression.
    Input: lists of values for each parameter, training points and observations, test points
    Output: tensor with loglikelihood entries for each possible parameter combination
    """
    
    param_grid = { 'ls_Phi': list(enumerate(ls_Phi_grid)),
                  'sigma_Phi' : list(enumerate(sigma_Phi_grid)),
                  'ls_A' : list(enumerate(ls_A_grid)),
                  'sigma_A' : list(enumerate(sigma_A_grid)),
                  'obs_noise' : list(enumerate(obs_noise_grid))}

    grid = ParameterGrid(param_grid)

    mat_ll = np.zeros((len(ls_Phi_grid), len(sigma_Phi_grid), len(ls_A_grid), len(sigma_A_grid), len(obs_noise_grid)))
    best = [0,0,0,0,0]
    ll = -100000
    itera = 0
    for params in grid:
        ll_new = hr.vectorfield_GP_Regression_helm(XY_train, UV_train, XY_test,
                                             ls_Phi = tf.constant(params['ls_Phi'][1], dtype=tf.float32), 
                                             sigma_Phi = tf.constant(params['sigma_Phi'][1], dtype=tf.float32),
                                             ls_A = tf.constant(params['ls_A'][1], dtype=tf.float32), 
                                             sigma_A = tf.constant(params['sigma_A'][1], dtype=tf.float32),
                                             obs_noise = tf.constant(params['obs_noise'][1], dtype=tf.float32),
                                             grid_search = True)[2]
    
        mat_ll[params['ls_Phi'][0], params['sigma_Phi'][0], params['ls_A'][0], params['sigma_A'][0], params['obs_noise'][0]] = ll_new
        if ll_new > ll:
                    ll = ll_new
                    best = [params['ls_Phi'][1],params['sigma_Phi'][1],params['ls_A'][1],params['sigma_A'][1], params['obs_noise'][1]]
                    print(params['ls_Phi'][1],
                          params['sigma_Phi'][1],
                          params['ls_A'][1],
                          params['sigma_A'][1],
                          params['obs_noise'][1], ' : ', ll)
        itera += 1
        if (itera % 50) == 0:
            print("Iteration: ", itera)
        
    np.savez_compressed("../params_opt/" + file_name, ll = mat_ll)

    return mat_ll 


def grid_search_mult_ls(ls_Phi1_grid, sigma_Phi1_grid, ls_A1_grid, sigma_A1_grid,
                        ls_Phi2_grid, sigma_Phi2_grid, ls_A2_grid, sigma_A2_grid,
                        obs_noise_grid, XY_train, UV_train, XY_test, file_name="marginal_ll_gs_helm_mult_ls"):
    
    """
    function to perform grid search of parameters for GP regression, with multiple lengthscales.
    Input: lists of values for each parameter, training points and observations, test points
    Output: tensor with loglikelihood entries for each possible parameter combination
    """
    
    param_grid = { 'ls_Phi1': list(enumerate(ls_Phi1_grid)),
                  'sigma_Phi1' : list(enumerate(sigma_Phi1_grid)),
                  'ls_A1' : list(enumerate(ls_A1_grid)),
                  'sigma_A1' : list(enumerate(sigma_A1_grid)),
                  'ls_Phi2': list(enumerate(ls_Phi2_grid)),
                  'sigma_Phi2' : list(enumerate(sigma_Phi2_grid)),
                  'ls_A2' : list(enumerate(ls_A2_grid)),
                  'sigma_A2' : list(enumerate(sigma_A2_grid)),
                  'obs_noise' : list(enumerate(obs_noise_grid))}

    grid = ParameterGrid(param_grid)

    mat_ll = np.zeros((len(ls_Phi1_grid), len(sigma_Phi1_grid), len(ls_A1_grid), len(sigma_A1_grid), len(ls_Phi2_grid), len(sigma_Phi2_grid), len(ls_A2_grid), len(sigma_A2_grid), len(obs_noise_grid)))
    best = [0,0,0,0,0,0,0,0,0]
    ll = -100000
    itera = 0
    for params in grid:
        ll_new = hr.vectorfield_GP_Regression_helm_mult_ls(XY_train, UV_train, XY_test,
                                             ls_Phi1 = tf.constant(params['ls_Phi1'][1], dtype=tf.float32), 
                                             sigma_Phi1 = tf.constant(params['sigma_Phi1'][1], dtype=tf.float32),
                                             ls_A1 = tf.constant(params['ls_A1'][1], dtype=tf.float32), 
                                             sigma_A1 = tf.constant(params['sigma_A1'][1], dtype=tf.float32),
                                             ls_Phi2 = tf.constant(params['ls_Phi2'][1], dtype=tf.float32), 
                                             sigma_Phi2 = tf.constant(params['sigma_Phi2'][1], dtype=tf.float32),
                                             ls_A2 = tf.constant(params['ls_A2'][1], dtype=tf.float32), 
                                             sigma_A2 = tf.constant(params['sigma_A2'][1], dtype=tf.float32),
                                             obs_noise = tf.constant(params['obs_noise'][1], dtype=tf.float32),
                                             grid_search = True)[2]
    
        mat_ll[params['ls_Phi1'][0], params['sigma_Phi1'][0], params['ls_A1'][0], params['sigma_A1'][0], params['ls_Phi2'][0], params['sigma_Phi2'][0], params['ls_A2'][0], params['sigma_A2'][0], params['obs_noise'][0]] = ll_new
        if ll_new > ll:
                    ll = ll_new
                    best = [params['ls_Phi1'][1],params['sigma_Phi1'][1],params['ls_A1'][1],params['sigma_A1'][1], params['ls_Phi2'][1],params['sigma_Phi2'][1],params['ls_A2'][1],params['sigma_A2'][1], params['obs_noise'][1]]
                    print(params['ls_Phi1'][1],
                          params['sigma_Phi1'][1],
                          params['ls_A1'][1],
                          params['sigma_A1'][1],
                          params['ls_Phi2'][1],
                          params['sigma_Phi2'][1],
                          params['ls_A2'][1],
                          params['sigma_A2'][1],
                          params['obs_noise'][1], ' : ', ll)
        itera += 1
        if (itera % 50) == 0:
            print("Iteration: ", itera)
        
    np.savez_compressed("../params_opt/" + file_name, ll = mat_ll)

    return mat_ll 
    


def grid_search_uv(ls_u_grid, sigma_u_grid, ls_v_grid, sigma_v_grid, obs_noise_grid, 
                XY_train, UV_train, XY_test, file_name="marginal_ll_gs_uv"):
    """
    function to perform grid search of parameters for GP regression.
    Input: lists of values for each parameter, training points and observations, test points
    Output: tensor with loglikelihood entries for each possible parameter combination
    """
    
    param_grid = {'ls_u': list(enumerate(ls_u_grid)),
                  'sigma_u' : list(enumerate(sigma_u_grid)),
                  'ls_v' : list(enumerate(ls_v_grid)),
                  'sigma_v' : list(enumerate(sigma_v_grid)),
                  'obs_noise' : list(enumerate(obs_noise_grid))}

    grid = ParameterGrid(param_grid)

    mat_ll = np.zeros((len(ls_u_grid), len(sigma_u_grid), len(ls_v_grid), len(sigma_v_grid), len(obs_noise_grid)))
    best = [0,0,0,0,0]
    ll = -100000
    itera = 0
    for params in grid:
        ll_new = hr.vectorfield_GP_Regression_uv(XY_train, UV_train, XY_test,
                                             ls_u = tf.constant(params['ls_u'][1], dtype=tf.float32), 
                                             sigma_u = tf.constant(params['sigma_u'][1], dtype=tf.float32),
                                             ls_v = tf.constant(params['ls_v'][1], dtype=tf.float32), 
                                             sigma_v = tf.constant(params['sigma_v'][1], dtype=tf.float32),
                                             obs_noise = tf.constant(params['obs_noise'][1], dtype=tf.float32))[2]
    
        mat_ll[params['ls_u'][0], params['sigma_u'][0], params['ls_v'][0], params['sigma_v'][0], params['obs_noise'][0]] = ll_new
        if ll_new > ll:
                    ll = ll_new
                    best = [params['ls_u'][1],params['sigma_u'][1],params['ls_v'][1],params['sigma_v'][1], params['obs_noise'][1]]
                    print(params['ls_u'][1],
                          params['sigma_u'][1],
                          params['ls_v'][1],
                          params['sigma_v'][1],
                          params['obs_noise'][1], ' : ', ll)
        itera += 1
        if (itera % 50) == 0:
            print("Iteration: ", itera)
        
    np.savez_compressed("../params_opt/" + file_name, ll = mat_ll)

    return mat_ll 



#plotting functions for the profile likelihood once the grid search is done
def plot_two_params(mat_ll, i, j):
    """
    Function for profile likelihood given two params/dimensions i and j.
    Assumption: j > i (e.g. j = 3rd param, i = 1st param)
    Input: matrix of likelihoods, param i, param j
    Output: profile likelihood matrix for grid search of i, j parameter
    """
    i_size = mat_ll.shape[i]
    j_size =  mat_ll.shape[j]
    mat = np.empty([i_size, j_size])
    for ii in range(i_size):
        for jj in range(j_size):
            mat_ll_j = np.take(mat_ll, [jj], axis=j)
            mat_ll_ij = np.take(mat_ll_j, [ii], axis=i)
            mat[ii,jj] = np.max(mat_ll_ij)
    return mat


def profile_likelihoods(mat_ll, params_name):
    profile_likelihoods = dict()
    for i, param_1 in enumerate(params_name):
        for j, param_2 in enumerate(params_name):
            if i >= j: continue
            profile_likelihoods[(param_1, param_2)] = plot_two_params(mat_ll, i, j)
    return profile_likelihoods

def plot_pl(params_grid, profile_likelihoods, param1, param2, f, ax, cmap, levels):
    cs = ax.contourf(params_grid[param1], params_grid[param2], profile_likelihoods[(param1, param2)].T,
                 levels = levels, cmap = cmap, extend = 'both')
    ax.set_xlabel(param1); ax.set_ylabel(param2)
    f.colorbar(cs, ax = ax)
    return