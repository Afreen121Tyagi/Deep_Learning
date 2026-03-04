from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# Create a 3D classification dataset
X, y = make_classification(n_samples=200, n_features=3, n_informative=3, n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, random_state=41, class_sep=2.5)


def step(z):
    return 1 if z > 0 else 0


def perceptron_weights_history(X, y, epochs=200, lr=0.1):
    Xb = np.insert(X, 0, 1, axis=1)  # add bias term
    weights = np.ones(Xb.shape[1])
    history = []
    accuracy_history = []
    n = Xb.shape[0]
    for i in range(epochs):
        j = np.random.randint(0, n)
        y_hat = step(np.dot(Xb[j], weights))
        weights = weights + lr * (y[j] - y_hat) * Xb[j]
        history.append(weights.copy())
        
        # compute accuracy on full dataset
        y_pred = np.array([step(np.dot(Xb[k], weights)) for k in range(n)])
        acc = np.mean(y_pred == y)
        accuracy_history.append(acc)
    
    return np.array(history), np.array(accuracy_history)


# run for more epochs so we can sample a few representative planes
weights_history, accuracy_history = perceptron_weights_history(X, y, epochs=1000, lr=0.1)

# find convergence point: when accuracy reaches max and stays stable
max_acc = np.max(accuracy_history)
convergence_idx = np.where(accuracy_history >= max_acc * 0.99)[0]
if len(convergence_idx) > 0:
    convergence_idx = convergence_idx[0]  # first frame at 99% of max accuracy
else:
    convergence_idx = len(weights_history) - 1


# Setup 3D plot with better styling
fig = plt.figure(figsize=(12, 9), facecolor='#1a1a1a')
ax = fig.add_subplot(111, projection='3d', facecolor='#0d0d0d')

# Plot data points with enhanced colors
scatter = ax.scatter(X[y==0, 0], X[y==0, 1], X[y==0, 2], 
                     c='#00D9FF', s=120, label='Class 0', alpha=0.8, edgecolors='white', linewidth=0.5)
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], 
          c='#FF006E', s=120, label='Class 1', alpha=0.8, edgecolors='white', linewidth=0.5)

# grid for the plane
xx_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 15)
yy_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 15)
xx, yy = np.meshgrid(xx_range, yy_range)

plane_surf = [None]
text_obj = [None]


def compute_plane(weights):
    # weights: [bias, w1, w2, w3] representing b + w1*x + w2*y + w3*z = 0
    b = weights[0]
    w1, w2, w3 = weights[1], weights[2], weights[3]
    eps = 1e-8
    if abs(w3) < eps:
        return np.full_like(xx, np.nan)
    zz = -(b + w1 * xx + w2 * yy) / w3
    return zz


def update(idx):
    """Draw plane corresponding to history index `idx`."""
    # remove previous surface if exists
    if plane_surf[0] is not None:
        try:
            plane_surf[0].remove()
        except Exception:
            pass
    
    zz = compute_plane(weights_history[idx])
    
    # Gradient color: red to green for progression
    progress = idx / convergence_idx if convergence_idx > 0 else 1.0
    color = plt.cm.RdYlGn(progress)
    
    plane = ax.plot_surface(xx, yy, zz, color=color, alpha=0.6, edgecolor='none', shade=True)
    plane_surf[0] = plane
    
    # Calculate accuracy for this epoch
    Xb = np.insert(X, 0, 1, axis=1)
    y_pred = np.array([step(np.dot(Xb[k], weights_history[idx])) for k in range(len(X))])
    acc = np.mean(y_pred == y)
    
    ax.set_title(f'Perceptron Training | Epoch: {idx+1} | Accuracy: {acc:.1%}', 
                fontsize=14, color='white', pad=20, weight='bold')
    
    return plane,


ax.set_xlabel('Feature 1', fontsize=11, color='white', weight='bold')
ax.set_ylabel('Feature 2', fontsize=11, color='white', weight='bold')
ax.set_zlabel('Feature 3', fontsize=11, color='white', weight='bold')
ax.view_init(elev=25, azim=45)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Styling
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('white')

# pick frames from start to convergence point, then repeat final frame
num_frames = 50
indices = np.linspace(0, convergence_idx, num_frames, dtype=int)
# Add final frame repeated 8 times (2 second pause at 300ms interval)
indices = np.concatenate([indices, [convergence_idx] * 8])

# use the subsampled indices as frame values directly
anim = FuncAnimation(fig,
                     update,
                     frames=indices,
                     interval=200,
                     repeat=True,
                     repeat_delay=1000)

# Save animation to GIF (requires pillow)
anim.save('perceptron_3d.gif', writer='pillow', dpi=80)
print("✅ Animation saved as 'perceptron_3d.gif'")
plt.show()