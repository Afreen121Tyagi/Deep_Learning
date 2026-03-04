from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=100, n_features=2, n_informative=1,n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=10)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap='viridis',s=120,edgecolors='k',linewidth=0.5)
plt.show()

def step(z):
    return 1 if z>0 else 0

def perceptron(X,y):
    
    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    for i in range(1000):
        j = np.random.randint(0,100)
        y_hat = step(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]
        
    return weights[0],weights[1:]


intercept_,coef_ = perceptron(X,y)

print(coef_)
print(intercept_)

m = -(coef_[0]/coef_[1])
b = -(intercept_/coef_[1])


x_input = np.linspace(-3,3,100)
y_input = m*x_input + b

plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='#FF00FF',linewidth=3)
plt.scatter(X[:,0],X[:,1],c=y,cmap='viridis',s=120,edgecolors='k',linewidth=0.5)
plt.ylim(-3,2)


def perceptron(X,y):
    m = []
    b = []

    X = np.insert(X,0,1,axis=1)
    weights = np.ones(X.shape[1])
    lr = 0.1
    
    
    for i in range(200):
        j = np.random.randint(0,100)
        y_hat = step(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]

        m.append(-(weights[1]/weights[2]))
        b.append(-(weights[0]/weights[2]))
        
    return m,b


m, b = perceptron(X,y)



# %matplotlib notebook
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation





fig, ax = plt.subplots(figsize=(10,5), facecolor='#111111')
x_i = np.arange(-3,3,0.1)
y_i = m[0]*x_i + b[0]
# use viridis map for points during animation
anim_scatter = ax.scatter(X[:,0],X[:,1],c=y,cmap='viridis',s=120,edgecolors='k',linewidth=0.5)
line, = ax.plot(x_i,y_i,color='#00FF66',linewidth=3)  # bright neon green
plt.ylim(-3,2)
ax.set_facecolor('#222222')
ax.tick_params(colors='white')


def update(i):
    label = 'epoch {0}'.format(i+1)
    line.set_ydata(m[i]*x_i + b[i])
    ax.set_xlabel(label, color='#FFFFFF')
    # return line,
anim = FuncAnimation(fig, update, repeat = True,frames=range(200), interval=100)
anim.save('perceptron_2d.gif', writer='pillow')
plt.show()