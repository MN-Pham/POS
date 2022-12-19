import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return (x-np.pi)**2 + (y-np.e)**2 + np.sin(3*x+0.75) + np.cos(4*y-2.13)

# constants to defind the size of the domain
size = 5

# constants for POS algorithm
num_particles = 20
num_iter = 20
w = 0.5
c1 = c2 = 0.2

# create random points with N(size, size^2) distribution
X = np.random.rand(2, num_particles) * size

# velocity with normal distribution
V = np.random.rand(2, num_particles) * 0.2

#X1 is the old array of particles
#X2 is the new array which was updated by V
def find_pbest(X1_, X2_):
    f1 = f(X1_[0], X1_[1])
    f2 = f(X2_[0], X2_[1])
    # if X2[i] is closer to the min point, replace X1[i] by X2[i]
    X1_[:, f2<f1] = X2_[:, f2<f1]
    return X1_

def find_gbest(X_):
    #find the index of the particle being closest to the min
    idx = f(X_[0], X_[1]).argmin()
    return X_[:, idx].reshape(-1,1) 

def find_V(V_, X_, w_, c1_, c2_, pbest_, gbest_):
    r = np.random.rand(2)
    V_ = w_*V_ + c1_*r[0]*(pbest_-X_) + c2_*r[1]*(gbest_-X_)
    return V_

pbest = X
gbest = find_gbest(X)

for i in range(num_iter):
    V = find_V(V, X, w, c1, c2, pbest, gbest)
    X_n = X+V
    pbest = find_pbest(X, X_n)
    gbest = find_gbest(X_n)
    X = X_n
        
    # Draw an image every 5 iterations and in the last iteration
    if int(i%5) == 0 or (i == num_iter-1):
        # Draw the background of the plot
        x, y = np.array(np.meshgrid(np.linspace(0,5,100), np.linspace(0,5,100)))
        z = f(x, y)
        x_min = x.ravel()[z.argmin()]
        y_min = y.ravel()[z.argmin()]
        fig, ax = plt.subplots(figsize=(8,6))
        fig.set_tight_layout(True)
        img = ax.imshow(z, extent=[0, size, 0, size], origin='lower', cmap='viridis', alpha=0.5)
        fig.colorbar(img, ax=ax)
        ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
        contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
        ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    
        p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
        p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
        gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
        ax.set_title("Iteration No." + str(i))

print("Location of the minimal point is:", find_gbest(X)[0].item(), ",", find_gbest(X)[1].item())
