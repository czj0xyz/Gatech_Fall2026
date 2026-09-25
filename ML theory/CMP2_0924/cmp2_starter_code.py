import numpy as np
from matplotlib import pyplot as plt
import math

def CreateHypothesisClass(grid_length):
    H = []
    for left in range(grid_length):
        for right in range(grid_length,left,-1):
            for bottom in range(grid_length):
                for top in range(grid_length,bottom,-1):
                    H.append( (left/grid_length,right/grid_length,bottom/grid_length,top/grid_length) )

    return H


def CreateDistribution(grid_length, Gaussians):
    sum_prob = 0
    x_distribution ={}
    y_distribution = {}
    for i in range(grid_length):
        for j in range(grid_length):
            x1 = (i+0.5)/grid_length
            x2 = (j+0.5)/grid_length
            positive_prob = 0
            negative_prob = 0
            for label, x1_mean, x2_mean, x1_std, x2_std in Gaussians:
                prob = np.exp( -((x1-x1_mean)/x1_std )**2/2 - ((x2-x2_mean)/x2_std )**2/2 )
                if label == 1: 
                    positive_prob += prob
                elif label == -1:
                    negative_prob += prob

            total_prob = positive_prob + negative_prob
            sum_prob += total_prob
            x_distribution[(x1,x2)] = total_prob
            y_distribution[(x1,x2)] = positive_prob / total_prob

    for x in x_distribution.keys():
        x_distribution[x] /= sum_prob

    return x_distribution, y_distribution



def Sample(distribution, n, seed):
    sampler = np.random.default_rng(seed=seed)
    data = []

    x_distribution, y_distribution = distribution
    input_set = list(x_distribution.keys())
    x_distribution_vector = np.array([x_distribution[x] for x in input_set])

    for input_idx in sampler.choice(np.arange(len(input_set)), n, replace=True, p=x_distribution_vector):
        if sampler.random() < y_distribution[input_set[input_idx]]:
            data.append((input_set[input_idx],1))
        else:
            data.append((input_set[input_idx],-1))
    return data

def Evaluate(h,x):
    if x[0] >= h[0] and x[0] <= h[1] and x[1] >= h[2] and x[1] <= h[3]:
        return 1
    else:
        return -1
    
def L(distribution, h):
    x_distribution, y_distribution = distribution
    error = 0
    for x in x_distribution.keys():
        if Evaluate(h,x) == 1:
            error += x_distribution[x] * (1-y_distribution[x])
        else:
            error += x_distribution[x] * y_distribution[x]
    return error
    
def L_hat(dataset, h):
    error_count = 0
    for x, y in dataset:
        if Evaluate(h,x) != y:
            error_count += 1
    return error_count / len(dataset)

def Get_h_hat(dataset, hypothesis_class):
    best_L_hat = 10
    best_h = None
    for h in hypothesis_class:
        if L_hat(dataset,h) < best_L_hat:
            best_L_hat = L_hat(dataset,h)
            best_h = h
    return best_h

def Get_h_star(distribution, hypothesis_class):
    best_L = 10
    best_h = None
    for h in hypothesis_class:
        if L(distribution,h) < best_L:
            best_L = L(distribution,h)
            best_h = h
    return best_h

def VisualizeDistribution(distribution, hypotheses_and_colors, fig_name="plot.png"):
    x_distribution, y_distribution = distribution
    input_set = list(x_distribution.keys())
    x1_vector = np.array([x[0] for x in input_set])
    x2_vector = np.array([x[1] for x in input_set])
    x_distribution_vector = np.array([x_distribution[x] for x in input_set])
    y_distribution_vector = np.array([y_distribution[x] for x in input_set])

    plt.figure()
    plt.scatter(x1_vector,x2_vector,c=y_distribution_vector,s=3000*x_distribution_vector,cmap="coolwarm")
    plt.colorbar()

    for h, color in hypotheses_and_colors:
        plt.plot((h[0],h[1],h[1],h[0],h[0]),(h[3],h[3],h[2],h[2],h[3]), c=color)

    plt.xlim((-0.01,1.01))
    plt.ylim((-0.01,1.01))
    plt.gca().set_aspect('equal')
    plt.savefig(fig_name)
    plt.close()

def VisualizeDataset(dataset, hypotheses_and_colors, fig_name="plot.png"):
    x_values = set([x for x,y in dataset])
    cmap = {-1: "b", 1: "r"}

    x1_values = []
    x2_values = []
    c_values = []

    plt.figure()
    for x_value in x_values:
        labels = [y for x,y in dataset if x==x_value]
        if len(labels) == 1:
            x1_values.append(x_value[0])
            x2_values.append(x_value[1])
            c_values.append( cmap[labels[0]] )
        else:
            radius = 0.003 + 0.002*len(labels)

            for idx, label in enumerate(labels):
                x1_values.append( x_value[0] + radius*math.cos(2*np.pi*idx/len(labels)) )
                x2_values.append( x_value[1] + radius*math.sin(2*np.pi*idx/len(labels)) )
                c_values.append( cmap[label] )

    plt.scatter(x1_values,x2_values,c=c_values,s=10)

    for h, color in hypotheses_and_colors:
        plt.plot((h[0],h[1],h[1],h[0],h[0]),(h[3],h[3],h[2],h[2],h[3]), c=color)

    plt.grid()
    plt.xlim((-0.01,1.01))
    plt.ylim((-0.01,1.01))
    plt.gca().set_aspect('equal')
    plt.savefig(fig_name)
    plt.close()

def Get_h_max(dataset, distribution, hypothesis_class):
    best = -np.inf
    best_h = None
    for h in hypothesis_class:
        if L(distribution,h)-L_hat(dataset,h) > best:
            best = L(distribution,h)-L_hat(dataset,h)
            best_h = h
    return best_h


grid_length = 20
Gaussians = [(1,0.2,0.5,0.1,0.2),
                 (1,0.8,0.6,0.05,0.05),
                 (-1,0,0,0.4,0.3)]
hypothesis_class = CreateHypothesisClass(grid_length) # preference for larger hypothesis boxes
distribution = CreateDistribution(grid_length, Gaussians)


## Part 1
# VisualizeDistribution(distribution,[],"part1_distribution.png")
# dataset = Sample(distribution,100,seed=0)
# VisualizeDataset(dataset,[],"part1_dataset.png")


## Part 2A
# for seed in range(5):
#     dataset = Sample(distribution,25,seed=seed)
#     h_star = Get_h_star(distribution, hypothesis_class)
#     h_hat = Get_h_hat(dataset, hypothesis_class)

#     hypotheses_and_colors = [(h_star,"black"),(h_hat,"green")]
#     VisualizeDistribution(distribution,hypotheses_and_colors,f"part2a_distribution{seed}.png")
#     VisualizeDataset(dataset,hypotheses_and_colors,f"part2a_dataset{seed}.png")

#     print()
#     print("seed",seed)
#     print("population error for best-in-class",L(distribution,h_star))
#     print("population error for ERM",L(distribution,h_hat))
#     print("empirical error for best-in-class",L_hat(dataset,h_star))
#     print("empirical error for ERM",L_hat(dataset,h_hat))


## Part 2B
# hypothesis_class = hypothesis_class[::-1]  # preference for smaller hypothesis boxes
# for seed in range(5):
#     dataset = Sample(distribution,25,seed=seed)
#     h_star = Get_h_star(distribution, hypothesis_class)
#     h_hat = Get_h_hat(dataset, hypothesis_class)

#     hypotheses_and_colors = [(h_star,"black"),(h_hat,"green")]
#     VisualizeDistribution(distribution,hypotheses_and_colors,f"part2b_distribution{seed}.png")
#     VisualizeDataset(dataset,hypotheses_and_colors,f"part2b_dataset{seed}.png")

#     print()
#     print("seed",seed)
#     print("population error for best-in-class",L(distribution,h_star))
#     print("population error for ERM",L(distribution,h_hat))
#     print("empirical error for best-in-class",L_hat(dataset,h_star))
#     print("empirical error for ERM",L_hat(dataset,h_hat))


# Part 3
for seed in range(10):
    dataset = Sample(distribution,25,seed=seed)
    h_star = Get_h_star(distribution, hypothesis_class)
    h_hat = Get_h_hat(dataset, hypothesis_class)
    h_max = Get_h_max(dataset, distribution, hypothesis_class)

    h_star_diff = L(distribution,h_star) - L_hat(dataset,h_star)
    h_hat_diff = L(distribution,h_hat) - L_hat(dataset,h_hat)
    h_max_diff = L(distribution,h_max) - L_hat(dataset,h_max)

    diff = []
    for h in hypothesis_class:
        diff.append(L(distribution,h) - L_hat(dataset,h))

    plt.figure()
    plt.hist(diff,bins=200,range=(-1,1))
    plt.axvline(h_star_diff, c="black")
    plt.axvline(h_hat_diff, c="green")
    plt.axvline(h_max_diff, c="magenta")
    # plt.plot([h_star_diff,h_star_diff],[0,1000],c="black")
    # plt.plot([h_hat_diff,h_hat_diff],[0,1000],c="green")
    # plt.plot([h_max_diff,h_max_diff],[0,1000],c="magenta")
    plt.savefig(f"part3_histogram{seed}.png")
    plt.close()

## Part 4
# print("n=25")
# for seed in range(5):
#     dataset = Sample(distribution,25,seed=seed)
#     h = Get_h_max(dataset, distribution, hypothesis_class)
#     print("seed =", seed, " LHS =",L(distribution,h)-L_hat(dataset,h))

# print("n=100")
# for seed in range(5):
#     dataset = Sample(distribution,100,seed=seed)
#     h = Get_h_max(dataset, distribution, hypothesis_class)
#     print("seed =",seed, " LHS =",L(distribution,h)-L_hat(dataset,h))

# delta = .05
# def calc_LHS_RHS(n):
#     dataset = Sample(distribution,n,seed=0)
#     h = Get_h_max(dataset, distribution, hypothesis_class)
#     LHS = L(distribution,h)-L_hat(dataset,h)
#     RHS = np.sqrt((np.log(len(hypothesis_class))+np.log(1/delta))*.5/n)
#     print(LHS, RHS)

# calc_LHS_RHS(25)
# calc_LHS_RHS(100)