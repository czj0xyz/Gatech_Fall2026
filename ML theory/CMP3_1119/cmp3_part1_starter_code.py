import numpy as np
import matplotlib.pyplot as plt

class GaussianProcessModel:
    def __init__(self, x_values, kernel, noise_sigma, points=[], seed=0):
        self.kernel = kernel
        self.x_values = x_values
        self.noise_sigma = noise_sigma
        self.points = points
        self.rng = np.random.default_rng(seed)

    def AddPoint(self, x, y):
        self.points.append((x, y))

    def Mean(self):
        if not self.points:
            return np.zeros_like(self.x_values)

        points_x = [x for x, y in self.points]

        K1 = self.KernelMatrix(points_x, points_x) + self.noise_sigma**2 * np.eye(len(self.points))
        K2 = self.KernelMatrix(points_x, self.x_values)

        return K2.T @ np.linalg.inv(K1) @ np.array([y for x, y in self.points])

    def Cov(self):
        if not self.points:
            return self.KernelMatrix(self.x_values, self.x_values) + self.noise_sigma**2 * np.eye(len(self.x_values))

        points_x = [x for x, y in self.points]
        K1 = self.KernelMatrix(points_x, points_x) + self.noise_sigma**2 * np.eye(len(self.points))
        K2 = self.KernelMatrix(points_x, self.x_values)
        K3 = self.KernelMatrix(self.x_values, self.x_values) + self.noise_sigma**2 * np.eye(len(self.x_values))

        return K3 - K2.T @ np.linalg.inv(K1) @ K2
    
    def Stddev(self):
        return np.sqrt(np.diag(self.Cov()))

    def KernelMatrix(self, X1, X2):
        K = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = self.kernel(x1, x2)
        return K
    
    def Sample(self):
        mean = self.Mean()
        cov = self.Cov()
        return self.rng.multivariate_normal(mean, cov)
    
def PlotFunctionsAndPoints(x_values, Ys, colors, points, title):
    plt.figure(figsize=(10, 6))
    for ys, color in zip(Ys, colors):
        plt.plot(x_values, ys, alpha=0.7, color=color)
    if points:
        px, py = zip(*points)
        plt.scatter(px, py, color='black')
    plt.title(title)
    plt.xlabel('x')
    plt.grid()
    plt.savefig(f'{title}.png')

def GaussianKernel(length_scale):
    def kernel(x, y):
        return np.exp(-0.5 * ((x - y) / length_scale) ** 2)
    return kernel

x_values = np.arange(0, 1.001, 0.001)
gp_model = GaussianProcessModel(x_values, GaussianKernel(length_scale=0.05), noise_sigma=0.0,seed=0)


## Sample from the prior
plot_functions = [gp_model.Sample() for _ in range(3)]
colors = len(plot_functions)*['blue']

plot_functions.append( gp_model.Mean())
colors.append('red')

PlotFunctionsAndPoints(x_values, plot_functions, colors, gp_model.points, title=f'After 0 Observations')

## Sample and condition on 3 observations
x_indices = [100,200,900]
f = gp_model.Sample()
for x_index in x_indices:
    gp_model.AddPoint(x_values[x_index], f[x_index])

plot_functions = [gp_model.Sample() for _ in range(3)]
colors = len(plot_functions)*['blue']

plot_functions.append( gp_model.Mean())
colors.append('red')

plot_functions.append(f)
colors.append('black')

PlotFunctionsAndPoints(x_values, plot_functions, colors, gp_model.points, title=f'After 3 Observations')



## Plotting the eigenvalues and eigenfunctions of the Kernel Matrix
K = gp_model.KernelMatrix(x_values, x_values)
eigvals, eigvecs = np.linalg.eig(K)

# Plot the eigenvalue spectrum
plt.figure(figsize=(10, 6))
plt.plot(eigvals, 'o-')
plt.title('Eigenvalues of the Kernel Matrix')
plt.yscale('log')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.grid()
plt.savefig('Eigenvalues_of_Kernel_Matrix.png')

# Plot the eigenfunctions corresponding to the largest eigenvalues
PlotFunctionsAndPoints(x_values, [eigvecs[:, i] for i in range(5)], ['red', 'orange', 'purple', 'blue', 'green'], [], title='Eigenvectors of the Kernel Matrix')
