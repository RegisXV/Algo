#Pso Algorithm
from __future__ import division
import random 
import math
import numpy as np
import matplotlib.pyplot as plt

#Rastringin Function
def rastringin(x):
    return 10*len(x) + np.sum(x**2 - 10*np.cos(2*math.pi*x))

#sphere Function
def sphere(x):
    return np.sum(x**2)

#Ackley Function
def ackley(x):
    return -20*np.exp(-0.2*np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2*math.pi*x))) + 20 + np.exp(1)

#Rosenbrock Function
def rosenbrock(x):
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

#Griewank Function
def griewank(x):
    return 1 + np.sum(x**2/4000) - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))

class Particle:
    def __init__(self, x0):
        self.position_i = np.array(x0, dtype=np.float64)  # Ensure position is float64
        self.velocity_i = np.random.uniform(-1, 1, len(x0))  # particle velocity
        self.pos_best_i = np.copy(self.position_i)  # best position individual
        self.err_best_i = float('inf')  # best error individual
        self.err_i = float('inf')  # error individual

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i:
            self.pos_best_i = np.copy(self.position_i)
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognitive constant
        c2 = 2  # social constant

        r1 = np.random.random(len(self.position_i))
        r2 = np.random.random(len(self.position_i))

        vel_cognitive = c1 * r1 * (self.pos_best_i - self.position_i)
        vel_social = c2 * r2 * (pos_best_g - self.position_i)
        self.velocity_i = w * self.velocity_i + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        self.position_i += self.velocity_i

        # adjust maximum position if necessary
        self.position_i = np.clip(self.position_i, bounds[:, 0], bounds[:, 1])

class PSO():
    def __init__(self, costFunc, costFuncName, x0, bounds, num_particles, maxiter):
        global num_dimensions

        num_dimensions = len(x0)
        err_best_g = float('inf')  # best error for group
        pos_best_g = np.zeros(num_dimensions)  # best position for group

        bounds = np.array(bounds, dtype=np.float64)  # Ensure bounds are float64

        # establish the swarm
        swarm = [Particle(x0) for _ in range(num_particles)]

        # begin optimization loop
        for i in range(maxiter):
            for particle in swarm:
                particle.evaluate(costFunc)

                # determine if current particle is the best (globally)
                if particle.err_i < err_best_g:
                    pos_best_g = np.copy(particle.position_i)
                    err_best_g = particle.err_i

            for particle in swarm:
                particle.update_velocity(pos_best_g)
                particle.update_position(bounds)

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)
        
        # plotting
        plt.figure()
        plt.plot(pos_best_g)
        plt.title(f'PSO Algorithm - {costFuncName}')
        plt.xlabel('Dimension')
        plt.ylabel('Best Position')
        plt.grid(True)
        plt.show()

# Run
initial = [5, 5]  # initial starting location [x1, x2...]
bounds = [(-10, 10), (-10, 10)]  # input bounds [(x1_min, x1_max), (x2_min, x2_max)...]

PSO(rastringin, "Rastringin", initial, bounds, num_particles=30, maxiter=200)
PSO(sphere, "Sphere", initial, bounds, num_particles=15, maxiter=100)
PSO(ackley, "Ackley", initial, bounds, num_particles=30, maxiter=200)
PSO(rosenbrock, "Rosenbrock", initial, bounds, num_particles=15, maxiter=100)
PSO(griewank, "Griewank", initial, bounds, num_particles=15, maxiter=100)
