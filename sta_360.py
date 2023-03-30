import matplotlib.pyplot as plt
import numpy as np
from math import factorial

def likelihood(y, theta):
    return ((5*theta)**y * np.exp(-5*theta)) / factorial(y)

def prior(theta):
    return 1/101

def marginal_likelihood(y):
    theta = np.linspace(0, 1, 101)
    return np.sum(likelihood(y, theta) * prior(theta))

def posterior(y, theta):
    return (likelihood(y, theta) * prior(theta)) / marginal_likelihood(y)

# Observation Y = 2

y = 2

# Theta values
theta = np.linspace(0, 1, 101)

# Plot the posterior probability
plt.stem(theta, posterior(y, theta))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta|y)$')
plt.title(r'Posterior Probability of $\theta$ given $y=2$')
plt.suptitle('Particles Emitted from a Substance')
plt.show()