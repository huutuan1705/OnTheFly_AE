import numpy as np
import matplotlib.pyplot as plt
import torch

# -------- Sinh dữ liệu --------
def generate_points(mode="uniform", n_points=100):
    if mode == "collapse":
        # Tập trung quanh góc 45° (pi/4) với nhiễu nhỏ
        angle = np.pi/4 + 0.05*np.random.randn(n_points)
    elif mode == "uniform":
        # Trải đều trên [0, 2π)
        angle = np.random.rand(n_points) * 2 * np.pi
    else:
        raise ValueError("Unknown mode")
    x = np.cos(angle)
    y = np.sin(angle)
    return np.stack([x, y], axis=1)

# -------- Tính Uniformity --------
def uniformity(z, nsamp=8192):
    z = torch.tensor(z, dtype=torch.float32)
    N = z.size(0)
    idx_i = torch.randint(0, N, (nsamp,))
    idx_j = torch.randint(0, N, (nsamp,))
    mask = idx_i != idx_j
    d2 = (z[idx_i[mask]] - z[idx_j[mask]]).pow(2).sum(dim=1)
    return torch.log(torch.exp(-2*d2).mean()).item()

# -------- Sinh và tính --------
points_c = generate_points("collapse", 100)
points_u = generate_points("uniform", 100)

uni_c = uniformity(points_c)
uni_u = uniformity(points_u)

print("Uniformity (collapse):", uni_c)
print("Uniformity (uniform):", uni_u)

# -------- Vẽ --------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Collapse
axes[0].scatter(points_c[:,0], points_c[:,1], color="red", alpha=0.7)
circle = plt.Circle((0,0), 1, color="black", fill=False, linestyle="--")
axes[0].add_artist(circle)
axes[0].set_title(f"Collapse\nUniformity = {uni_c:.2f}")
axes[0].set_xlim([-1.2, 1.2])
axes[0].set_ylim([-1.2, 1.2])
axes[0].set_aspect('equal')

# Uniform
axes[1].scatter(points_u[:,0], points_u[:,1], color="blue", alpha=0.7)
circle = plt.Circle((0,0), 1, color="black", fill=False, linestyle="--")
axes[1].add_artist(circle)
axes[1].set_title(f"Uniform\nUniformity = {uni_u:.2f}")
axes[1].set_xlim([-1.2, 1.2])
axes[1].set_ylim([-1.2, 1.2])
axes[1].set_aspect('equal')

plt.show()
