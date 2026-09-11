import math
import matplotlib.pyplot as plt
import numpy as np


def BinarySearch(func, low, high, threshold, tolerance):
    # Given a continuous function func and a interval [low,high], returns a value that is within tolerance of a point x where func(x)=threshold
    assert(low <= high)
    func_low = func(low)
    func_high = func(high)
    if func_low <= func_high:
        assert(func_low <= threshold and func_high >= threshold)
        return BinarySearchRecursive(lambda x: func(x) - threshold,low,high,tolerance)
    else:
        assert(func_low >= threshold and func_high <= threshold)
        return BinarySearchRecursive(lambda x: -(func(x) - threshold),low,high,tolerance)
    
def BinarySearchRecursive(func,low,high,tolerance):
    mid = (low + high)/2

    if high - low <= 2*tolerance:
        return mid
    
    if func(mid) > 0:
        return BinarySearchRecursive(func,low,mid,tolerance)
    else:
        return BinarySearchRecursive(func,mid,high,tolerance)
    
def InvertBound(Bound, n, k, delta=0.05, tolerance=10**-6):
    if Bound(n,0,True,k) >= delta/2:
        lower_bound=0
    else:
        lower_bound = BinarySearch(lambda p: Bound(n,p,True,k),0,1,delta/2,tolerance)
    
    if Bound(n,1,False,k) >= delta/2:
        upper_bound=1
    else:
        upper_bound = BinarySearch(lambda p: Bound(n,p,False,k),0,1,delta/2,tolerance)
    
    return lower_bound, upper_bound

def Binomial_Prob(n,p,greater_than,k):
    assert(isinstance(n, int))
    assert(0 <= k and k <= n)
    assert(0 <= p and p <= 1)
    if not isinstance(k,int):
        if greater_than:
            k = n-int(n-k)
        else:
            k = int(k)

    if greater_than and k==0:
        return 1
    if not greater_than and k==n:
        return 1
                    
    if greater_than:
        return 1 - Binomial_Prob(n,p,False,k-1)

    if k > n//2:
        return 1 - Binomial_Prob(n,1-p,False,n-k-1)

    if p==0:
        return 1
    if p==1:
        return 0
        
    running_log_prob = n*math.log(1-p)
    running_sum = math.exp( running_log_prob )

    for i in range(1,k+1):
        running_log_prob += math.log( (n+1-i)/i * p/(1-p))
        running_sum += math.exp( running_log_prob)

    return running_sum

def DKL(p,q):
    assert(0 <= p and p <= 1)
    assert(0 <= q and q <= 1)

    if p==q:
        return 0

    if p==0:
        return math.log(1/(1-q))
    if p==1:
        return math.log(1/q)

    return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))


## Part 1

def Hoeffding_Bound(n,p,greater_than,k):
    if(greater_than == True):
        return 1. if n*p > k else np.exp(-2*n*(k/n-p)**2)#eps=k/n-p
    else: return 1. if n*p < k else np.exp(-2*n*(p-k/n)**2)#eps=p-k/n
    # raise NotImplementedError

def Bernstein_Bound(n,p,greater_than,k):
    # raise NotImplementedError
    eps = np.abs(k/n - p)
    if(p == 0. and k == 0.):return 1.
    if(p == 1. and k == n): return 1.
    return 1. if not (greater_than^(n*p>k)) else np.exp(-(n*eps**2*.5)/((1-p)*p+eps/3))

def Chernoff_Bound(n,p,greater_than,k):
    # raise NotImplementedError
    if(p==1.): return float(n >= k if greater_than else n <= k)
    elif(p==0.): return float(0 >= k if greater_than else 0 <= k)
    return 1. if not (greater_than^(n*p>k)) else np.exp(-n*DKL(k/n,p))

## Part 2

bounds = [("Binomial",Binomial_Prob),
          ("Hoeffding",Hoeffding_Bound), 
          ("Bernstein",Bernstein_Bound), 
          ("Chernoff",Chernoff_Bound)]

### Part 2.A
n=1000
p=0.05
k=75
for bound in bounds:
    bound_name, bound_function = bound
    print(bound_function, bound_function(n,p,True,k))

### Part 2.B
n=1000
k=100
for n, k in [(1000,100), (1000,500), (1000,10), (1000,0), (100,10), (10000,1000)]:
    for bound in bounds:
        bound_name, bound_function = bound
        print(bound_function, InvertBound(bound_function, n,k))
        # l,r = InvertBound(bound_function, n,k)
        # print(r-l)
    # print()


### Part 2.C
def CountMisses(Bound, n, p, num_samples, delta=0.05):
    bounds_for_k = [InvertBound(Bound, n,k,delta) for k in range(n+1)]
    misses=0
    for i in range(num_samples):
        k = np.random.binomial(n,p)
        lower_bound, upper_bound = bounds_for_k[k]
        if p < lower_bound or p > upper_bound:
            misses+=1
    return misses/num_samples
  
n=1000
p=0.1
num_samples=10**6
for bound in bounds:
    bound_name, bound_function = bound
    print(bound_function, CountMisses(bound_function, n,p, num_samples))


### Part 2.D
def PlotInvertedBound(Bound, n_values, k_values, x_axis_values, label, delta=0.05,tolerance=10**-6):
    bounds = [InvertBound(Bound, n,k,delta=delta,tolerance=tolerance) for n,k in zip(n_values,k_values)]
    lower_bounds, upper_bounds = zip(*bounds)

    plt.plot(x_axis_values, lower_bounds, label=label)
    plt.plot(x_axis_values, upper_bounds, label=label)

# n_values = 1001*[1000]
# k_values = list(range(1001))
# x_axis_values = k_values

n_values = list(i*10 for i in range(1,1001))
k_values = list(range(1,1001))
x_axis_values = n_values

plt.plot(x_axis_values,[k/n for n,k in zip(n_values,k_values)],c="k")
for bound in bounds:
    bound_name, bound_function = bound
    PlotInvertedBound(bound_function,n_values,k_values,x_axis_values,bound_name)

plt.legend()
plt.savefig("plot2.png")