import numpy as np
import torch

def GBC_all(f_s: np.ndarray, y: np.ndarray, current_labels):

    # print(current_labels[-2:])

    # TRY FLOAT64. 64 BIT PRECISION.
    # PCA
    # ADD THE DIAGONAL.

    # PRIORITIZING THE SCAFFOLDING

    perclass = dict.fromkeys(current_labels)

    for c in perclass:
        
        print(f_s, y, f_s[y==c])
        f_s_c = f_s[y == c, :].detach()
        f_s_c = f_s_c.type(torch.DoubleTensor)
        print(f_s_c)
        N_c = f_s_c.shape[0]

        mu_c = sum(f_s_c) / N_c
        mu_c = mu_c.unsqueeze(1)

        # cov_c = sum((f_s_c[i, :] - mu_c) @ (f_s_c[i, :] - mu_c).T for i in range(N_c)) / (N_c-1)
        torch_cov_c = torch.cov(f_s_c.T)
        print(f"Torch determinant of class {c}: {torch.det(torch_cov_c)}")
        perclass[c] = [mu_c, torch_cov_c] # should this be cov_c?

    gbc = 0
    vals = list(perclass.values())

    for i, j in zip(range(len(vals)), range(1, len(vals))):
        bc = torch.exp(-bhattacharyya_dist(vals[i], vals[j]))
        gbc -= bc

    return gbc

def bhattacharyya_dist(c_i, c_j):

    # print(c_i, c_j)

    mu_c_i, mu_c_j = c_i[0], c_j[0]
    cov_c_i, cov_c_j = c_i[1], c_j[1]

    cov = (cov_c_i + cov_c_j) / 2

    cov += torch.eye(cov.shape[0]) * 1e-3

    # print(torch.det(cov) / np.sqrt(torch.det(cov_c_i) * torch.det(cov_c_j)))
    # print(torch.det(cov_c_i), torch.det(cov_c_j))

    D_b = (mu_c_i - mu_c_j).T @ torch.inverse(cov) @ (mu_c_i - mu_c_j) / 8 + np.log(torch.det(cov) / np.sqrt(torch.det(cov_c_i) * torch.det(cov_c_j))) / 2
    
    # print("Sigma: ", cov, "\nSigma inverse", torch.inverse(cov))
    # print("Determinant of top 5:", cov[:5, :5])
    # print("Determinant: ", torch.det(cov))

    # print((np.sqrt(torch.det(cov_c_i) * torch.det(cov_c_j))))
    
    return D_b