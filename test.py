import torch

import torch, numpy as np, matplotlib.pyplot as plt

@torch.no_grad()
def uniformity(x: torch.Tensor, t: float = 2.0):
    sim = x @ x.T                         # cosine sim
    N = x.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=x.device)
    sqdist = (2 - 2 * sim)[mask]          # ||u-v||^2 = 2 - 2cos(u,v)
    return torch.log(torch.exp(-t * sqdist).mean()).item()

def pca_2d_numpy(x: np.ndarray):
    X = x - x.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T

# ---------- 2) Chuẩn bị sketch vectors từ list (20,64) ----------
def pool_sketch_list(sketch_list, mode="mean", device=None):
    pooled = []
    for s in sketch_list:
        s = s.to(device) if device is not None else s
        if mode == "mean":
            v = s.mean(dim=0)             # (64,)
        elif mode == "max":
            v, _ = s.max(dim=0)           # (64,)
        else:
            raise ValueError("mode should be 'mean' or 'max'")
        pooled.append(v.unsqueeze(0))
    return torch.cat(pooled, dim=0)

# ---------- 3) Vẽ scatter PCA 2D + hiển thị uniformity ----------
def visualize_uniformity_scatter(sketch_list, photo_features,
                                 pool="mean", t=2.0, save_path=None):
    """
    sketch_list: list of (20,64)
    photo_features: (N,64)
    """
    device = photo_features.device if isinstance(photo_features, torch.Tensor) else None

    # (a) Sketch -> (M,64)
    S = pool_sketch_list(sketch_list, mode=pool, device=device)
    # (b) Photo -> (N,64) (đảm bảo tensor)
    P = photo_features if isinstance(photo_features, torch.Tensor) else torch.as_tensor(photo_features)

    # (c) Uniformity (tính trên không gian gốc, trước PCA)
    u_s = uniformity(S, t=t)
    u_p = uniformity(P, t=t)
    u_joint = uniformity(torch.cat([S, P], dim=0), t=t)

    print(f"[Uniformity] sketch={u_s:.4f}  photo={u_p:.4f}  joint={u_joint:.4f}  (t={t}, pool={pool})")

    # (d) PCA 2D - convert 64D to 2D
    X = torch.cat([S, P], dim=0).cpu().numpy()
    Z = pca_2d_numpy(X)
    M = S.size(0)
    Zs, Zp = Z[:M], Z[M:]

    # (e) Scatter + box hiện số uniformity
    plt.figure(figsize=(6,6))
    plt.scatter(Zs[:,0], Zs[:,1], s=10, marker='o', alpha=0.7, label=f'Sketch ({pool})')
    plt.scatter(Zp[:,0], Zp[:,1], s=14, marker='x', alpha=0.8, label='Photo')
    plt.title("Uniformity view — PCA 2D")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(loc='best')
    txt = f"Uniformity (t={t}):\n  sketch = {u_s:.4f}\n  photo  = {u_p:.4f}\n  joint  = {u_joint:.4f}"
    plt.gcf().text(0.02, 0.02, txt, fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle="round", alpha=0.2))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.show()

# -------------------------------
# Ví dụ dữ liệu
N = 100          # số ảnh
M_per = 3       # số sketch mỗi ảnh
D = 64          # số chiều
M = N * M_per   # tổng số sketch

# Ảnh: (N, D), normalize sẵn
photo_features = torch.randn(N, D)
photo_features = photo_features / photo_features.norm(dim=1, keepdim=True)

# Sketch list: M phần tử, mỗi phần tử (20, D), normalize sẵn
sketch_list = []
for _ in range(M):
    s = torch.randn(20, D)
    s = s / s.norm(dim=1, keepdim=True)
    sketch_list.append(s)

# Gọi hàm visualize
visualize_uniformity_scatter(sketch_list, photo_features,
                             pool="mean", t=2.0,
                             save_path="uniformity_scatter_demo.png")

