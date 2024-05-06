import numpy as np
import torch

def GBC(f_s: np.ndarray, y: np.ndarray, current_labels):

    # print(current_labels[-2:])

    # TRY FLOAT64. 64 BIT PRECISION.
    # PCA
    # ADD THE DIAGONAL.

    # PRIORITIZING THE SCAFFOLDING

    perclass = dict.fromkeys(current_labels[-2:])

    print(perclass)
    for c in perclass:
        
        print(y, c)
        f_s_c = f_s[y == c, :].detach()
        f_s_c = f_s_c.type(torch.DoubleTensor)
        N_c = f_s_c.shape[0]

        mu_c = sum(f_s_c) / N_c
        mu_c = mu_c.unsqueeze(1)

        # print("Debug:")
        # print(f"N_c = {N_c}")
        # print(f"f_s_c shape = {f_s_c.shape}")
        # print(f"f_s shape = {f_s.shape}")
        # cov_c = sum((f_s_c[i, :] - mu_c) @ (f_s_c[i, :] - mu_c).T for i in range(N_c)) / (N_c-1)
        torch_cov_c = torch.cov(f_s_c.T)
        # print(torch_cov, cov_c)
        # print(torch_cov.shape, cov_c.shape)
        print(f"Torch determinant of class {c}: {torch.det(torch_cov_c)}")
        # print(f"Torch log det" , torch.logdet(torch_cov))
        # print(f"Orig determinant: {torch.det(cov_c)}")
        # print(f"Orig log det: {torch.logdet(cov_c)}")

        # print("Mean cov", mu_c, cov_c)
        perclass[c] = [mu_c, torch_cov_c] # should this be cov_c?

    gbc = 0
    # for i in range(len(perclass.keys())):
    #     for j in range(i+1, len(perclass.keys())):
    #         bc = torch.exp(-bhattacharyya_dist(perclass[i], perclass[j]))

    #         print(bc)
    #         gbc -= bc

    # for i in range(len(perclass.keys())):
    #     for j in range(i+1, len(perclass.keys())):
    vals = list(perclass.values())
    bc = torch.exp(-bhattacharyya_dist(vals[0], vals[1]))

    # print(bc)
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