import numpy as np
import matplotlib.pyplot as plt

def ExcessMSE(eigenvalues, sigma, num_train, num_test, seed=0):
    """
    Calculate the excess mean squared error (MSE) for a linear regression model
    using the provided eigenvalues of the data covariance matrix, noise variance,
    number of training samples, and number of test samples.

    Parameters:
    eigenvalues (np.ndarray): Eigenvalues of the data covariance matrix.
    sigma (float): Label noise standard deviation.
    num_train (int): Number of training samples.
    num_test (int): Number of test samples.
    seed (int): Random seed for reproducibility.

    Returns:
    float: The calculated excess MSE.
    """
    d = len(eigenvalues)
    print(f"Dimension: {d}")
    
    theta_star = np.zeros(d)
    theta_star[0] = 1.0  # Assume the true parameter has a value of 1 in the first dimension

    rng = np.random.default_rng(seed)
    train_x = np.multiply(rng.normal(size=(num_train, d)), np.sqrt(eigenvalues))
    train_y = train_x @ theta_star + rng.normal(scale=sigma, size=num_train)

    test_x = np.multiply(rng.normal(size=(num_test, d)), np.sqrt(eigenvalues))
    test_y = test_x @ theta_star + rng.normal(scale=sigma, size=num_test)

    if num_train >= d:
        theta_hat = np.linalg.inv(train_x.T @ train_x) @ train_x.T @ train_y
    else:
        theta_hat = train_x.T @ np.linalg.inv(train_x @ train_x.T) @ train_y

    test_predictions = test_x @ theta_hat
    excess_mse = np.mean((test_predictions - test_y) ** 2) - sigma**2

    return excess_mse

# ## Vary covariance spectrum
# d_array = [10,20,30,40,50,60,70,80,90,100,200,400,800,1600,3200]
# max_d = max(d_array)
# n=100
# n_test = 10000
# sigma = 0.1
# mean_excess_mse_dict = {}
# for eig_type in ['constant','harmonic']:
# # for eig_type in ['constant','harmonic','spike1','spike2','spike3','spike4']:
# # for eig_type in ['constant','harmonic','exponential80','exponential90','exponential95','exponential98','exponential99']:
#     if eig_type == 'constant':
#         eigs = np.ones(max_d)
#     elif eig_type == 'harmonic':
#         eigs = np.float_power(np.arange(1, max_d + 1), -1)
#     elif eig_type == 'exponential95':
#         eigs = np.float_power(0.95, np.arange(0, max_d))
#     elif eig_type == 'exponential90':
#         eigs = np.float_power(0.90, np.arange(0, max_d))
#     elif eig_type == 'exponential80':
#         eigs = np.float_power(0.80, np.arange(0, max_d))
#     elif eig_type == 'exponential98':
#         eigs = np.float_power(0.98, np.arange(0, max_d))
#     elif eig_type == 'exponential99':
#         eigs = np.float_power(0.99, np.arange(0, max_d))
#     elif eig_type == 'spike1':
#         eigs = (1/10**1)*np.ones(max_d)
#         eigs[:10] = 1.0  # Spike at the first 10 dimensions
#     elif eig_type == 'spike2':
#         eigs = (1/10**2)*np.ones(max_d)
#         eigs[:10] = 1.0  # Spike at the first 10 dimensions
#     elif eig_type == 'spike3':
#         eigs = (1/10**3)*np.ones(max_d)
#         eigs[:10] = 1.0  # Spike at the first 10 dimensions
#     elif eig_type == 'spike4':
#         eigs = (1/10**4)*np.ones(max_d)
#         eigs[:10] = 1.0  # Spike at the first 10 dimensions

#     mean_excess_mse = []
#     for d in d_array:
#         excess_mse_values = []
#         for seed in range(10,20):
#             excess_mse = ExcessMSE(eigs[:d], sigma=sigma, num_train=n, num_test=n_test, seed=seed)
#             excess_mse_values.append(excess_mse)
#         mean_excess_mse.append(np.mean(excess_mse_values))
#     mean_excess_mse_dict[eig_type] = mean_excess_mse

# plt.figure(figsize=(8,6))
# plt.xscale('log')
# plt.yscale('log')
# for eig_type, mean_excess_mse in mean_excess_mse_dict.items():
#     plt.plot(d_array, mean_excess_mse, label=eig_type, marker='o')
# plt.plot(d_array,np.ones_like(d_array),c='k',label='Always predict 0')
# plt.xlabel('Dimension (d)')
# plt.ylabel('Excess MSE')
# plt.title('Excess MSE vs Dimension')
# plt.legend() 
# plt.savefig('excess_mse_vary_eigs.png')


## Vary Sigma
d_array = [200,400,800,1600,3200]
max_d = max(d_array)
eigs = np.float_power(np.arange(1, max_d + 1), -1)
n=100
n_test = 10000

mean_excess_mse_dict = {}
for sigma2 in [0,0.01,0.02,0.03,0.04,0.05]:
    mean_excess_mse = []
    for d in d_array:
        excess_mse_values = []
        for seed in range(10):
            excess_mse = ExcessMSE(eigs[:d], sigma=np.sqrt(sigma2), num_train=n, num_test=n_test, seed=seed)
            excess_mse_values.append(excess_mse)
        mean_excess_mse.append(np.mean(excess_mse_values))
    mean_excess_mse_dict[sigma2] = mean_excess_mse

plt.figure(figsize=(8,6))
plt.xscale('log')
for sigma2, mean_excess_mse in mean_excess_mse_dict.items():
    plt.plot(d_array, mean_excess_mse, label=f'sigma^2={sigma2}', marker='o')
plt.xlabel('Dimension (d)')
plt.ylabel('Excess MSE')
plt.title('Excess MSE vs Dimension')
plt.legend() 
plt.savefig('excess_mse_vary_sigma.png')