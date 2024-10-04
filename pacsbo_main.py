import torch
import torch.nn as nn
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tqdm import tqdm
import time
import warnings
import copy
# import concurrent
import multiprocessing
import tikzplotlib
import pickle
import dill
from matplotlib.patches import Ellipse
import torch.multiprocessing as mp
from scipy.special import comb
from plot import plot_1D, plot_2D_contour, plot_1D_SafeOpt_with_sets, plot_gym, plot_gym_together
from ground_truth_experiment import ground_truth_experiment
import gym
import sys
sys.path.insert(1,  './vision-based-furuta-pendulum-master')
from gym_brt.envs import QubeBalanceEnv, QubeSwingupEnv
from gym_brt.control.control import QubeHoldControl, QubeFlipUpControl
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from IPython import embed as IPS


class net(nn.Module):
    def __init__(self, input_size, output_size):
        super(net, self).__init__()
        self.l1 = nn.Linear(input_size, 15)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(15, 25)
        self.l3 = nn.Linear(25, 12)
        self.l4 = nn.Linear(12, output_size)

    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        output = self.relu(output)
        output = self.l3(output)
        output = self.relu(output)
        output = self.l4(output)
        return output


def NN_return_prediction(h_RKHS_data, vi_frac_data):
    # no exponential extrapolation used!
    max_len = 14
    h_RKHS_input = [0]*max_len
    vi_frac_input = [0]*max_len
    h_RKHS_input[(max_len - len(h_RKHS_data)):] = h_RKHS_data  # fill up with non-zero elements
    vi_frac_input[(max_len - len(vi_frac_data)):] = vi_frac_data  # fill up with non-zero elements

    # What about the coefficients of the random RKHS functions
    input_tensor = torch.tensor(h_RKHS_input + vi_frac_input).to(torch.float32)
    model_load = net(len(input_tensor), 1)
    # model_load.load_state_dict(torch.load('NN-hardware_final_final'))
    model_load.load_state_dict(torch.load('NN_final'))
    model_load.eval()
    RKHS_norm_prediction = model_load(input_tensor)
    return RKHS_norm_prediction.item()


class GPRegressionModel(gpytorch.models.ExactGP):  # this model has to be build "new"
    def __init__(self, train_x, train_y, noise_std, n_devices=1, output_device=torch.device('cpu'), lengthscale=0.1):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(noise_std**2)
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        # self.kernel = gpytorch.kernels.rbf_kernel.RBFKernel()
        self.kernel.lengthscale = lengthscale
        # self.base_kernel.lengthscale.requires_grad = False; somehow does not work
        if output_device.type != 'cpu':
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                self.kernel, device_ids=range(n_devices), output_device=output_device)
        else:
            self.covar_module = self.kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def reshape_with_extra_dims(tensor, num_dims):
    # Calculate the number of extra dimensions needed
    extra_dims = [1] * (num_dims - tensor.dim())  # Adjust extra dimensions based on the tensor's shape
    # Reshape the tensor with extra dimensions
    reshaped_tensor = tensor.unsqueeze(*extra_dims).float()  # Convert tensor to floating-point
    return reshaped_tensor


def convert_to_hashable(item):
    if isinstance(item, (torch.Tensor, np.ndarray)):
        return tuple(map(tuple, item.tolist()))
    elif isinstance(item, tuple):
        return tuple(convert_to_hashable(i) for i in item)
    else:
        return item


def compute_X_plot(n_dimensions, points_per_axis):
    X_plot_per_domain = torch.linspace(0, 1, points_per_axis)
    X_plot_per_domain_nd = [X_plot_per_domain] * n_dimensions
    X_plot = torch.cartesian_prod(*X_plot_per_domain_nd).reshape(-1, n_dimensions)
    return X_plot


def initial_safe_samples(gt, num_safe_points):
    fX = gt.fX
    num_safe_points = num_safe_points
    # sampling_logic = torch.logical_and(fX > np.quantile(fX, 0.8), fX < np.quantile(fX, 0.99))
    # sampling_logic = fX > gt.safety_threshold
    sampling_logic = torch.logical_and(fX > np.quantile(fX, 0.4), fX < np.quantile(fX, 0.50))
    random_indices_sample = torch.randint(high=X_plot[sampling_logic].shape[0], size=(num_safe_points,))
    X_sample = X_plot[sampling_logic][random_indices_sample]
    Y_sample = fX[sampling_logic][random_indices_sample] + torch.tensor(np.random.normal(loc=0, scale=noise_std, size=X_sample.shape[0]), dtype=torch.float32)
    return X_sample, Y_sample


class ground_truth():
    def __init__(self, num_center_points, X_plot, RKHS_norm):
        def fun(kernel, alpha):
            return lambda X: kernel(X.reshape(-1, self.X_center.shape[1]), self.X_center).detach().numpy() @ alpha
        # For ground truth
        self.X_plot = X_plot
        self.RKHS_norm = RKHS_norm
        random_indices_center = torch.randint(high=self.X_plot.shape[0], size=(num_center_points,))
        self.X_center = self.X_plot[random_indices_center]
        alpha = np.random.uniform(-1, 1, size=self.X_center.shape[0])
        self.kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        self.kernel.lengthscale = 0.1
        RKHS_norm_squared = alpha.T @ self.kernel(self.X_center, self.X_center).detach().numpy() @ alpha
        alpha /= np.sqrt(RKHS_norm_squared)/RKHS_norm  # scale to RKHS norm
        self.f = fun(self.kernel, alpha)
        self.fX = torch.tensor(self.f(self.X_plot), dtype=torch.float32)
        self.safety_threshold = np.quantile(self.fX, 0.3)  # np.quantile(self.fX, np.random.uniform(low=0.15, high=0.5))

    def conduct_experiment(self, x, noise_std):
        return torch.tensor(self.f(x) + np.random.normal(loc=0, scale=noise_std, size=1), dtype=x.dtype)

    def local_RKHS_norm(self, lb, ub, X_plot_local=None):
        nugget_factor = 1e-4
        # Returns RKHS norm of ground truth on local domain between lb and ub
        if X_plot_local is None:
            local_gt_indices = torch.all(torch.logical_and(self.X_plot >= lb, self.X_plot <= ub), axis=1)
            if sum(local_gt_indices) > 10000:  # the problem is that the matrix gets very high dimensional and we cannot explicitly compute it
                subset_boolean = torch.randperm(sum(local_gt_indices)) < 10000
                X_local = self.X_plot[local_gt_indices][subset_boolean]
                fX_local = self.fX[local_gt_indices][subset_boolean]
            else:
                X_local = self.X_plot[local_gt_indices]
                fX_local = self.fX[local_gt_indices]
        else:  # use the local X_plots
            if X_plot.shape[0] > 10000:
                X_local = X_plot_local[torch.randperm(X_plot_local.shape[0]) < 10000]
                fX_local = torch.tensor(self.f(X_local), dtype=X_local.dtype)
            else:
                X_local = X_plot_local
                fX_local = torch.tensor(self.f(X_local), dtype=X_local.dtype)
        K_local = self.kernel(X_local, X_local).evaluate() + torch.eye(X_local.shape[0])*nugget_factor  # add a small nugget factor
        local_RKHS_norm_value = torch.sqrt(fX_local.reshape(1, -1) @ torch.inverse(K_local) @ fX_local.reshape(-1, 1)).flatten()  # inverse computation takes a lot of time but does not need to be done often. True RKHS norm given eigentlich once. And again also just for training
        while not local_RKHS_norm_value > 0:
            nugget_factor *= 10
            K_local = self.kernel(X_local, X_local).evaluate() + torch.eye(X_local.shape[0])*nugget_factor  # add a small nugget factor
            local_RKHS_norm_value = torch.sqrt(fX_local.reshape(1, -1) @ torch.inverse(K_local) @ fX_local.reshape(-1, 1)).flatten()  # inverse computation takes a lot of time but does not need to be done often. True RKHS norm given eigentlich once. And again also just for training
        return local_RKHS_norm_value  # This is for training only anyways. Local kernel interpolation approach will not be accurate. 


class PACSBO():
    def __init__(self, delta_confidence, intermediate_domain_size, noise_std, tuple_ik, X_plot, X_sample,
                Y_sample, safety_threshold, exploration_threshold, gt, compute_local_X_plot, compute_all_sets=False):
        def compute_X_plot_locally(n_dimensions, points_per_axis, lb, ub):
            X_plot = []
            for i in range(n_dimensions):
                X_plot_per_domain = torch.linspace(lb[i], ub[i], points_per_axis)
                X_plot.append(X_plot_per_domain)
            X_plot = torch.cartesian_prod(*X_plot).reshape(-1, n_dimensions)
            return X_plot
        self.compute_all_sets = compute_all_sets
        self.gt = gt  # at least for toy experiments it works like this.
        self.exploration_threshold = exploration_threshold
        self.delta_confidence = delta_confidence
        self.X_plot = X_plot
        self.noise_std = noise_std
        self.n_dimensions = X_plot.shape[1]
        self.best_lower_bound_local = -np.infty
        self.safety_threshold = safety_threshold
        self.tuple = tuple_ik
        self.lambda_bar = max(self.noise_std, 1)
        if tuple_ik != (-1, -1):  # "key" for the global domain
            if tuple_ik == (0, 0):  # convex hull
                self.lb = min(X_sample)
                self.ub = max(X_sample)
            elif tuple_ik == (1, 1): # intermediate domain
                self.lb = min(X_sample) - intermediate_domain_size  # the domain is pre-scaled to [0,1]^n
                self.ub = max(X_sample) + intermediate_domain_size
                self.lb[self.lb < 0] = 0  # clipping
                self.ub[self.ub > 1] = 1  # clipping
            sample_indices = torch.all(torch.logical_and(X_sample >= self.lb, X_sample <= self.ub), axis=1)  # good command. axis=1 operates on the rows. If all elements in one row are true, it returns True
            self.x_sample = X_sample[sample_indices].clone().detach()
            self.y_sample = Y_sample[sample_indices].clone().detach()
            if not compute_local_X_plot:
                self.discr_domain = X_plot[torch.all(torch.logical_and(self.X_plot >= self.lb, self.X_plot <= self.ub), axis=1)]  # good command
            else:
                self.discr_domain = compute_X_plot_locally(n_dimensions=self.X_plot.shape[1], points_per_axis=int(np.round(self.X_plot.shape[0]**(1/self.X_plot.shape[1]))),
                                                           lb=self.lb, ub=self.ub)
        else:
            self.lb = torch.tensor([0]*X_plot.shape[1])
            self.ub = torch.tensor([1]*X_plot.shape[1])
            self.x_sample = X_sample
            self.y_sample = Y_sample
            self.discr_domain = X_plot

    def compute_model(self, dict_reuse_GPs, gpr):
        if convert_to_hashable(self.x_sample) in dict_reuse_GPs.keys():  # use some rounding to accuracy of discretized domain
            self.model, self.K = dict_reuse_GPs[convert_to_hashable(self.x_sample)]
        else:
            if Furuta:
                self.model = gpr(train_x=self.x_sample, train_y=self.y_sample, noise_std=self.noise_std, lengthscale=0.2)
            else:
                self.model = gpr(train_x=self.x_sample, train_y=self.y_sample, noise_std=self.noise_std, lengthscale=0.1)
            self.K = self.model(self.x_sample).covariance_matrix
            # model.train()
            dict_reuse_GPs[convert_to_hashable(self.x_sample)] = [self.model, self.K]
        # return model

    def compute_mean_var(self):
        self.model.eval()
        self.f_preds = self.model(self.discr_domain)  # the X value will also be somehow hidden inside this f_preds
        self.mean = self.f_preds.mean
        self.var = self.f_preds.variance

    def compute_confidence_intervals_training(self, dict_local_RKHS_norms={}):
        if self.tuple in dict_local_RKHS_norms:
            self.B = dict_local_RKHS_norms[self.tuple]
        else:
            self.B = self.compute_RKHS_norm_true()
            dict_local_RKHS_norms[self.tuple] = self.B
        self.compute_beta()
        self.lcb = self.mean - self.beta*torch.sqrt(self.var)  # we have to use standard deviation instead of variance
        self.ucb = self.mean + self.beta*torch.sqrt(self.var)
        return dict_local_RKHS_norms

    def compute_confidence_intervals_evaluation(self, gamma_PAC=None, m_PAC=None, alpha_bar=None, PAC=True, RKHS_norm_guessed=None, N_hat=100):  # PAC is a boolean that decides whether we are in the outer loop or inner loop
        # Always compute the PAC RKHS norm
        if RKHS_norm_guessed is None:
            self.B = NN_return_prediction(self.RKHS_norm_mean_function_list, self.vi_frac_list)  # only place where we need the NN
            if PAC:
                list_random_RKHS_norms = []
                # list_random_RKHS_functions = []
                # N_hat = int(max(torch.round((torch.max(self.ub-self.lb))*500), len(self.y_sample) + 10))
                print(f'We have N_hat={N_hat} for cube {self.tuple}. Getting PAC bounds now.')
                x_interpol = self.x_sample
                y_interpol = self.y_sample
                for _ in tqdm(range(m_PAC)):
                    X_c = (torch.min(self.discr_domain) - torch.max(self.discr_domain))*torch.rand(N_hat, self.x_sample.shape[1]) + torch.max((self.discr_domain))
                    X_c_tail = X_c[x_interpol.shape[0]:]
                    X_c[:self.x_sample.shape[0]] = x_interpol  # this will be hard with local X_plot
                    alpha_tail = -2*alpha_bar*torch.rand(N_hat-len(y_interpol), 1) + alpha_bar
                    y_tail = self.model.kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
                    y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
                    # matrix inversion with nugget factor
                    alpha_head = torch.inverse(self.model.kernel(x_interpol, x_interpol).evaluate()+torch.eye(len(y_interpol))*1e-3) @ y_head
                    alpha = torch.cat((alpha_head, alpha_tail))
                    # if counter < 30:
                    #     random_RKHS_function = self.model.kernel(self.discr_domain, X_c).evaluate() @ alpha
                    #     list_random_RKHS_functions.append(random_RKHS_function)
                    # elif len(self.y_sample) > 3:
                    #     print(123)
                    #     plt.figure()
                    #     for random_RKHS_function in list_random_RKHS_functions:
                    #         plt.plot(self.discr_domain, random_RKHS_function.detach().numpy(), alpha=0.1, color='blue')
                    #     plt.scatter(self.x_sample, self.y_sample, color='black', s=100, label='Samples')

                    random_RKHS_norm = torch.sqrt(alpha.T @ self.model.kernel(X_c, X_c).evaluate() @ alpha)
                    list_random_RKHS_norms.append(random_RKHS_norm)
                numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
                maximum_coefficients = torch.cat((alpha_head, alpha_bar*torch.ones(N_hat-len(alpha_head), 1)))
                maximum_possible_RKHS_norm = torch.sqrt(maximum_coefficients.T @ torch.ones(N_hat, N_hat) @ maximum_coefficients)
                minimum_possible_RKHS_norm = torch.sqrt(Y_sample.reshape(1, -1)@self.model.kernel(X_sample,X_sample) @ Y_sample)  # kernel interpolation
                b_Hoeffding = maximum_possible_RKHS_norm.detach().numpy()
                a_Hoeffding = minimum_possible_RKHS_norm.detach().numpy()
                self.B = max(self.B, np.mean(numpy_list) + np.sqrt(np.log(2/gamma_PAC)*(b_Hoeffding-a_Hoeffding)**2/(2*m_PAC)).item())  # conservative

        elif RKHS_norm_guessed is not None:
            self.B = RKHS_norm_guessed
        self.compute_beta()
        self.lcb = self.mean - self.beta*torch.sqrt(self.var)  # we have to use standard deviation instead of variance
        self.ucb = self.mean + self.beta*torch.sqrt(self.var)

    def compute_safe_set(self):
        self.S = self.lcb > self.safety_threshold  # version without "Lipschitz constant"; as programmed in classical SafeOpt

        # Auxiliary objects of potential maximizers M and potential expanders G
        self.G = self.S.clone()
        self.M = self.S.clone()

    def maximizer_routine(self, best_lower_bound_others):
        self.M[:] = False  # initialize
        self.max_M_var = 0  # initialize
        if not torch.any(self.S):  # no safe points
            return
        self.best_lower_bound_local = max(self.lcb[self.S])
        self.M[self.S] = self.ucb[self.S] >= max(best_lower_bound_others, self.best_lower_bound_local)
        self.M[self.M.clone()] = (self.ucb[self.M] - self.lcb[self.M]) > self.exploration_threshold
        if not torch.any(self.M):
            return
        self.max_M_var = torch.max(self.ucb[self.M] - self.lcb[self.M])
        self.max_M_ucb = torch.max(self.ucb[self.M])

    def expander_routine(self):
        self.G[:] = False  # initialize
        if not torch.any(self.S) or torch.all(self.S):  # no safe points or all of them are safe points
            return
        # no need to consider points in M
        if self.compute_all_sets:  # for visualization; introductory example
            s = self.S.clone()
        else:
            s = torch.logical_and(self.S, ~self.M)
            s[s.clone()] = (self.ucb[s] - self.lcb[s]) > self.max_M_var
            s[s.clone()] = (self.ucb[s] - self.lcb[s]) > self.exploration_threshold  # only sufficiently uncertain.
        # still same size as the safe set! We are just over-writing the positive ones
        if not torch.any(s):
            return
        potential_expanders = self.discr_domain[s]
        unsafe_points = self.discr_domain[~self.S]
        kernel_distance = self.compute_kernel_distance(potential_expanders, unsafe_points)
        ucb_expanded = self.ucb[s].unsqueeze(1).expand(-1, kernel_distance.size(1))
        s[s.clone()] = torch.any(ucb_expanded - self.B*kernel_distance > self.safety_threshold, dim=1)
        # or go with for loop
        # boolean_expander = ~s[s.clone()]  # assume that all are NOT expanders and go in the loop
        # for i in range(len(potential_expanders)):
        #     potential_expander = potential_expanders[i]
        #     for unsafe_point in unsafe_points:
        #         if self.ucb[s][i] - self.compute_kernel_distance(potential_expander, unsafe_point) > self.safety_threshold:
        #             boolean_expander[i] = True
        #             break  # we only need one!  
        # s[s.clone()] = boolean_expander  # update the potential expanders on whether they can potentially expand to an unsafe point            
        self.G = s

    def compute_beta(self):
        # Bound from Fiedler et al. 2021
        # matrix_expression = self.lambda_bar/self.noise_std*self.K + self.lambda_bar*torch.eye(self.x_sample.shape[0])
        # self.beta = self.B + torch.sqrt(self.noise_std*(torch.log(torch.det(matrix_expression))-2*torch.log(torch.tensor(self.delta_confidence))))

        # Fiedler et al. 2024 Equation (7); based on Abbasi-Yadkori 2013
        # inside_log = 1/self.delta_confidence*torch.det(torch.eye(self.x_sample.shape[0]) + (1/self.noise_std*self.K)) # TODO: fix mistake
        # self.beta = self.B + torch.sqrt(2*self.noise_std*torch.log(inside_log))
        inside_log = torch.det(torch.eye(self.x_sample.shape[0]) + (1/self.noise_std*self.K))
        inside_sqrt = self.noise_std*torch.log(inside_log) - (2*self.noise_std*torch.log(torch.tensor(self.delta_confidence)))
        self.beta = self.B + torch.sqrt(inside_sqrt)

    def compute_RKHS_norm_true(self):
        return self.gt.local_RKHS_norm(lb=self.lb, ub=self.ub, X_plot_local=self.discr_domain) if self.tuple != (-1,-1) else self.gt.RKHS_norm

    def compute_kernel_distance(self, x, x_prime):  # let us try whether it works without reshaped!
        '''
        k(x,x)+k(x^\prime,x^\prime)-k(x,x^\prime)-k(x^\prime,x)=2-2k(x,x^\prime)
        This holds for all radial kernels with output variance 1, i.e., k(x,x)\equiv 1.
        Both of which are true for our case.
        We have this setting and we exploit it.
        '''
        # print('Before kernel operation')
        if self.model.kernel.__class__.__name__ != 'MaternKernel' and self.model.kernel.__class__.__name__ != 'RBFKernel':
            raise Exception("Current implementation only works with radial kernels.")
        matrix_containing_kernel_values = self.model.kernel(x, x_prime).evaluate()  # here we can have problems with the size of the matrix
        # print('After kernel operation')
        return torch.sqrt(2-2*matrix_containing_kernel_values)

    def save_data_for_RNN_training(self, dict_mean_RKHS_norms, dict_recip_variances, x_last_iteration):  # We can also just do it for the cube from which we sample, and of course global. But let's see.
        if convert_to_hashable(self.tuple) not in dict_mean_RKHS_norms.keys():  # RKHS norm only defined by the samples. We always start at the beginning.
            alpha = torch.inverse(self.K+self.noise_std**2*torch.eye(self.K.shape[0])) @ self.y_sample
            self.RKHS_norm_mean_function_list = [torch.sqrt(alpha.reshape(1, -1) @ self.K @ alpha.reshape(-1, 1)).flatten()]  # RKHS norm of the mean.
            dict_mean_RKHS_norms[self.tuple] = self.RKHS_norm_mean_function_list

            # TODO: check whether the discretization is correct like this, also with the nD case
            variance_integral = sum(self.var)/2*(self.n_dimensions/self.discr_domain.shape[0])
            self.vi_frac_list = [1/variance_integral]
            dict_recip_variances[self.tuple] = self.vi_frac_list
        elif x_last_iteration is None:  # just return the dictionaries
            pass
        elif torch.all(torch.logical_and(x_last_iteration >= self.lb, x_last_iteration <= self.ub)):  # the list already exists and the new point is inside the sub-domain. If the new point is not inside, we do not care to add data
            alpha = torch.inverse(self.K+self.noise_std**2*torch.eye(self.K.shape[0])) @ self.y_sample
            RKHS_norm_mean_function = torch.sqrt(alpha.reshape(1, -1) @ self.K @ alpha.reshape(-1, 1)).flatten()
            dict_mean_RKHS_norms[self.tuple].append(RKHS_norm_mean_function)
            self.RKHS_norm_mean_function_list = dict_mean_RKHS_norms[self.tuple]

            variance_integral = sum(self.var)/2*(self.n_dimensions/self.discr_domain.shape[0])
            dict_recip_variances[self.tuple].append(1/variance_integral)
            self.vi_frac_list = dict_recip_variances[self.tuple]
        else:  # Already in the list and new point is not influencing the sub-domain
            self.vi_frac_list = dict_recip_variances[self.tuple]
            self.RKHS_norm_mean_function_list = dict_mean_RKHS_norms[self.tuple]
        return dict_mean_RKHS_norms, dict_recip_variances


def run_PACSBO(args):
    training = args[-1]
    if training:  # boolean whether we are training or not
        hyperparameters, num_iterations, X_plot, RKHS_norm = args[:-1]
        global_approach = False
        noise_std, delta_confidence, exploration_threshold, intermediate_domain_size, compute_local_X_plot = hyperparameters
    if not training:
        hyperparameters, num_iterations, X_sample, Y_sample, gt, X_plot, global_approach, compute_local_X_plot = args[:-1]
        run_type = hyperparameters[-1]
        if run_type == 'SafeOpt':
            noise_std, delta_confidence, exploration_threshold, B, compute_all_sets, _ = hyperparameters
            intermediate_domain_size = None
        else:
            compute_all_sets = False
            noise_std, delta_confidence, alpha_bar, m_PAC, gamma_PAC,\
            exploration_threshold, intermediate_domain_size, _ = hyperparameters
    # progress_bar = tqdm(total=num_iterations)
    global_cube_list = []
    convex_cube_list = []
    intermediate_cube_list = []
    dict_mean_RKHS_norms = {}
    dict_recip_variances = {}
    if training:
        run_type = 'PACSBO'
        dict_local_RKHS_norms = {}
        list_training = []  # we can do that a posteriori
        gt = ground_truth(num_center_points=np.random.choice(range(600, 1000)), X_plot=X_plot, RKHS_norm=RKHS_norm)  # cannot pickle this object
        X_sample_init, Y_sample_init = initial_safe_samples(gt=gt, num_safe_points=num_safe_points)
        X_sample = X_sample_init.clone()
        Y_sample = Y_sample_init.clone()
        # print(RKHS_norm)
    if not global_approach:
        interesting_domains = {tuple([-1, -1]), tuple([0, 0]), tuple([1, 1])}  # PACSBO: only three interesting domains
    elif global_approach:
        interesting_domains = {tuple([-1, -1])}

    x_new_last_iteration = torch.tensor([-torch.inf for _ in range(n_dimensions)])  # init
    best_lower_bound_others = -np.infty  # init
    skip_global_domain = False  # init
    while X_sample.shape[0] <= num_iterations:  # unchangeable for-loop
        try:
            del chosen_cube  # just delete it completely
        except NameError:
            pass
        # best_lower_bound_others_old = max(best_lower_bound_others, -np.infty) if not global_approach else -np.infty  # such that we do not return infinity at the end.
        best_lower_bound_others = -np.infty
        max_uncertainty_interesting = 0  # max uncertainty of interesting domain
        dict_reuse_GPs = {}
        current_interesting_domains = interesting_domains.copy()
        if (-1, -1) in current_interesting_domains:  # we should iterate with the global domain
            current_interesting_domains.remove((-1, -1))
            domains_to_iterate_through = [(-1, -1), *current_interesting_domains]
        else:
            domains_to_iterate_through = [(-1, -1), *current_interesting_domains]  # global domain not interesting
        if skip_global_domain:
            domains_to_iterate_through.remove((-1, -1))
        for (i, k) in domains_to_iterate_through:  # start off with global domain
            skip_global_domain = False  # only valid when starting the while loop and we want to skip the global domain for next round.
            try:
                del cube  # just delete it completely
            except NameError:
                pass
            cube = PACSBO(delta_confidence=delta_confidence, intermediate_domain_size=intermediate_domain_size, noise_std=noise_std, tuple_ik=(i, k), X_plot=X_plot, X_sample=X_sample,
                            Y_sample=Y_sample, safety_threshold=gt.safety_threshold, exploration_threshold=exploration_threshold, gt=gt,
                            compute_local_X_plot=compute_local_X_plot, compute_all_sets=compute_all_sets)  # all samples that we currently have
            cube.compute_model(dict_reuse_GPs, gpr=GPRegressionModel)
            cube.compute_mean_var()
            if run_type == 'PACSBO':
                dict_mean_RKHS_norms, dict_recip_variances = cube.save_data_for_RNN_training(dict_mean_RKHS_norms, dict_recip_variances, x_new_last_iteration)  # RKHS norm and reciprocal covariance integral
            if training:
                dict_local_RKHS_norms = cube.compute_confidence_intervals_training(dict_local_RKHS_norms=dict_local_RKHS_norms)
            else:
                if run_type == 'PACSBO':
                    cube.compute_confidence_intervals_evaluation(gamma_PAC=gamma_PAC, m_PAC=m_PAC, alpha_bar=alpha_bar, PAC=True, N_hat=100)
                elif run_type == 'SafeOpt':
                    cube.compute_confidence_intervals_evaluation(RKHS_norm_guessed=B)
            cube.compute_safe_set()
            cube.maximizer_routine(best_lower_bound_others=best_lower_bound_others)
            cube.expander_routine()
            if cube.best_lower_bound_local > best_lower_bound_others:
                best_lower_bound_others = cube.best_lower_bound_local
            if not torch.any(torch.logical_or(cube.M, cube.G)):
                if cube.tuple in interesting_domains:
                    interesting_domains.remove(cube.tuple)
            else:
                max_uncertainty_interesting_local = max((cube.ucb - cube.lcb)[torch.logical_or(cube.M, cube.G)])
                x_new_current = cube.discr_domain[torch.logical_or(cube.M, cube.G)][torch.argmax(cube.var[torch.logical_or(cube.M, cube.G)])]
                if not torch.any(torch.all(X_sample == x_new_current, axis=1)) and max_uncertainty_interesting_local > max_uncertainty_interesting:
                    max_uncertainty_interesting = max_uncertainty_interesting_local
                    chosen_tuple = cube.tuple
                    chosen_cube = cube
                    x_new = x_new_current
                elif torch.any(torch.all(X_sample == x_new_current, axis=1)) and cube.tuple in interesting_domains:  # double removing strategy.
                    interesting_domains.remove(cube.tuple)
            if (i, k) == (0, 0):
                convex_cube_list.append(cube)
            elif (i, k) == (1, 1):
                intermediate_cube_list.append(cube)
            elif (i, k) == (-1, -1):
                global_cube_list.append(cube)
            
        if not training and run_type == 'PACSBO':
            # Now with PAC bounds!
            try:  # there is a chosen cube
                auxx = chosen_cube.tuple
            except:  # there is no chosen cube
                if not training:
                    print('PACSBO terminated! There is no input that we can/want to sample next.')  # this can happen.
                break

            dict_local_RKHS_norms = chosen_cube.compute_confidence_intervals_evaluation(gamma_PAC, m_PAC, alpha_bar, PAC=True, N_hat=100)
            chosen_cube.compute_safe_set()
            chosen_cube.maximizer_routine(best_lower_bound_others=best_lower_bound_others)
            chosen_cube.expander_routine()
            if torch.any(torch.logical_or(chosen_cube.M, chosen_cube.G)):
                x_new_current = chosen_cube.discr_domain[torch.logical_or(chosen_cube.M, chosen_cube.G)][torch.argmax(chosen_cube.var[torch.logical_or(chosen_cube.M, chosen_cube.G)])]
                if not torch.any(torch.all(X_sample == x_new_current, axis=1)):
                    x_new = x_new_current
                    print(f'The chosen cube is {chosen_tuple} and the input is {x_new}, with PAC RKHS norm {chosen_cube.B}.')
            if not torch.any(torch.logical_or(chosen_cube.M, chosen_cube.G)) or torch.any(torch.all(X_sample == x_new_current, axis=1)):
                if len(interesting_domains) == 0:  # but this should not happen...
                    if not training:
                        print('PACSBO terminated! There is no input that we can/want to sample next.')
                    break
                if chosen_cube.tuple in interesting_domains:
                    interesting_domains.remove(chosen_cube.tuple)
                    print('Skipping this domain after PAC check')
                    if chosen_cube.tuple == (-1, -1):
                        skip_global_domain = True
                    x_new_last_iteration = None if torch.any(x_new_last_iteration > np.infty) else x_new_last_iteration
                    continue
        else:
            try:
                # Auxiliary action but no print
                aux = copy.deepcopy(x_new)
            except:
                print(f'{run_type} terminated! There is no input that we can/want to sample next.')
                break
        if not training or training:
            pass
        y_new = gt.conduct_experiment(x=x_new, noise_std=noise_std)
        if y_new < gt.safety_threshold:
            if training or run_type == 'SafeOpt':
                warnings.warn('Sampled unsafe point!')
            else:
                raise Exception('Sampled unsafe point!')
        X_sample = torch.cat((X_sample, x_new.unsqueeze(0)), dim=0)
        Y_sample = torch.cat((Y_sample, y_new), dim=0)
        if Furuta:  # always save in hardware; you never know what happens
            with open('furuta_hardware_X_sample.pickle', 'wb') as handle:
                pickle.dump(X_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('furuta_hardware_Y_sample.pickle', 'wb') as handle:
                pickle.dump(Y_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Saved data. We currently gathered {len(Y_sample)} samples.')

        # Which sub-domain changed through this new sample?
        if not global_approach:  # SysDO: new sample influences every domain
            interesting_domains = {tuple([-1, -1]), tuple([0, 0]), tuple([1, 1])}
        x_new_last_iteration = copy.deepcopy(x_new)
        del x_new  # There is no x_new for the next iteration
    if training:
        list_training = []
        for key in dict_mean_RKHS_norms.keys():
            list_training.append([dict_mean_RKHS_norms[key], dict_recip_variances[key], dict_local_RKHS_norms[key]])
        return list_training  # all values that we got/need from ONE random RKHS function
    if not training:
        return X_sample, Y_sample, [convex_cube_list, intermediate_cube_list, global_cube_list], gt.safety_threshold


if __name__ == '__main__':

    # Hyperparameters
    Furuta = False  # set to True to conduct policy parameter optimization on the Furuta pendulum
    training = False
    mujoco = False
    if training and mujoco:
        raise Exception('Cannot get PACSBO training runs with mujoco')
    noise_std = 0.01
    delta_confidence = 0.1  # normal SafeOpt routine
    gamma_PAC = 0.1  # PAC bounds
    num_safe_points = 3
    num_iterations = 12  # number of total points at the end
    exploration_threshold = 0.1  # let us start with that, maybe we should decrease to 0. But let's see
    n_dimensions = 1
    points_per_axis = 1000

    # Initialize PACSBO
    X_plot = compute_X_plot(n_dimensions, points_per_axis)
    intermediate_domain_size = 0.1  # 10% enlargement of the convex hull
    compute_all_sets = False
    reproduce_experiments = True  # set to True if you want to reproduce the experiments

    # For training
    if training:
        compute_local_X_plot = False
        hyperparameters = [noise_std, delta_confidence, exploration_threshold, intermediate_domain_size, compute_local_X_plot]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            parallel = False  # somehow parallel does not work yet
            number_of_random_RKHS_function = 1000
            # num_iterations, X_plot, hyperparameters, RKHS_norm, training=True, gt=None, X_sample=None, Y_sample=None, global_approach=False
            task_input = [(hyperparameters, num_iterations, X_plot, np.random.uniform(0.5, 30), training) for _ in range(number_of_random_RKHS_function)]
            if parallel:
                # with mp.Pool(processes=1) as pool:
                with mp.Pool() as pool:
                    collected_training = pool.map(run_PACSBO, task_input)
            else:  # especially important for debugging
                collected_list_training = []
                for task in tqdm(task_input):
                    list_training = run_PACSBO(task)
                    collected_list_training.append(list_training)
        with open('1D_training_data.pickle', 'wb') as handle:
            pickle.dump(collected_list_training, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Training finished!')
    # For "evaluating"
    if not training and not Furuta:
        gamma_PAC = 0.1  # probability PAC bounds
        m_PAC = 1000  # number of random RKHS function created for PAC bounds
        alpha_bar = 1
        RKHS_norm = 2
        gt = ground_truth(num_center_points=1000, X_plot=X_plot, RKHS_norm=RKHS_norm)
        X_sample_init, Y_sample_init = initial_safe_samples(gt=gt, num_safe_points=num_safe_points)
        X_sample = X_sample_init.clone()
        Y_sample = Y_sample_init.clone()
        if reproduce_experiments:
            noise_std = 0.01
            with open('1D_toy_experiments/gt.pickle', 'rb') as handle:
                gt = dill.load(handle)
            with open('1D_toy_experiments/X_sample.pickle', 'rb') as handle:
                X_sample = dill.load(handle)
            with open('1D_toy_experiments/Y_sample.pickle', 'rb') as handle:
                Y_sample = dill.load(handle)
        if not reproduce_experiments:  # not reproduce_experiments:
            with open('1D_toy_experiments/gt.pickle', 'wb') as handle:
                dill.dump(gt, handle)
            with open('1D_toy_experiments/X_sample.pickle', 'wb') as handle:
                dill.dump(X_sample, handle)
            with open('1D_toy_experiments/Y_sample.pickle', 'wb') as handle:
                dill.dump(Y_sample, handle)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if reproduce_experiments:
                RKHS_norm = gt.RKHS_norm
            compute_local_X_plot = False
            run_type = 'SafeOpt'
            global_approach = True

            # B = RKHS_norm/5
            # hyperparameters = [noise_std, delta_confidence, exploration_threshold, B, compute_all_sets, run_type]
            # X_sample_SO_under, Y_sample_SO_under, [convex_cube_list_under, intermediate_cube_list_under, global_cube_list_under], _ = \
            #     run_PACSBO(args=[hyperparameters, num_iterations, X_sample, Y_sample, gt, X_plot, global_approach, compute_local_X_plot, training])
            # plot_1D(X_plot, gt.fX, title='SafeOpt under', safety_threshold=gt.safety_threshold,
            #         cube_list=[convex_cube_list_under, intermediate_cube_list_under, global_cube_list_under], plot_type='beginning', save=False)
            # plot_1D(X_plot, gt.fX, title='SafeOpt under', safety_threshold=gt.safety_threshold,
            #         cube_list=[convex_cube_list_under, intermediate_cube_list_under, global_cube_list_under], plot_type='end', save=False)

            B = RKHS_norm*5
            hyperparameters = [noise_std, delta_confidence, exploration_threshold, B, compute_all_sets, run_type]
            X_sample_SO_over, Y_sample_SO_over, [convex_cube_list_over, intermediate_cube_list_over, global_cube_list_over], _ = run_PACSBO(args=[hyperparameters, num_iterations, X_sample, Y_sample, gt, X_plot, global_approach, compute_local_X_plot, training])
            plot_1D(X_plot, gt.fX, title='SafeOpt over', safety_threshold=gt.safety_threshold,
                    cube_list=[convex_cube_list_over, intermediate_cube_list_over, global_cube_list_over], plot_type='beginning', save=False)
            plot_1D(X_plot, gt.fX, title='SafeOpt over', safety_threshold=gt.safety_threshold,
                    cube_list=[convex_cube_list_over, intermediate_cube_list_over, global_cube_list_over], plot_type='end', save=False)


            compute_local_X_plot = True
            run_type = 'PACSBO'
            hyperparameters = [noise_std, delta_confidence, alpha_bar, m_PAC, gamma_PAC,
                                exploration_threshold, intermediate_domain_size, run_type]
            global_approach = False
            X_sample_pbo, Y_sample_pbo, [convex_cube_list_pbo, intermediate_cube_list_pbo, global_cube_list_pbo], safety_threshold = \
                run_PACSBO(args=[hyperparameters, num_iterations, X_sample, Y_sample, gt, X_plot, global_approach, compute_local_X_plot, training])
            if n_dimensions == 1:
                plot_1D(X_plot=X_plot, fX=gt.fX, title='PACSBO', safety_threshold=safety_threshold,
                        cube_list=[convex_cube_list_pbo, intermediate_cube_list_pbo, global_cube_list_pbo], plot_type='beginning', save=False)
                plot_1D(X_plot=X_plot, fX=gt.fX, title='PACSBO', safety_threshold=safety_threshold,
                        cube_list=[convex_cube_list_pbo, intermediate_cube_list_pbo, global_cube_list_pbo], plot_type='end', save=False)
            elif n_dimensions == 2:
                plot_2D_contour(X_plot, gt.fX, X_sample_pbo, Y_sample=Y_sample_pbo, safety_threshold=gt.safety_threshold, title='PACSBO 2D', levels=10, save=False)
