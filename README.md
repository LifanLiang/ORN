# ORN
Use OR-gate Network (ORN) to model the impacts of somatic mutations on cancer transcriptomics

This implementation requires numba, numpy, and scipy installed. Alternatively, install Annaconda.

# Example usage
After running the ORN.py, use function _ORN_ to execute the algorithm.
The arguments for _ORN_ include:
- deg: the binary matrix of differential expression status, each row should be a sample, and each column should be a gene
- mut: the binary event matrix of somatic alterations, each row should be a sample. The number of samples should the same as deg.
- pathway_size: the number of pathways.
- alpha_X: the alpha prior for U (the relationship matrix between mut and path), default 1.0
- beta_X: the beta prior for U (the relationship matrix between mut and path), default 1.0
- alpha_Z: the alpha prior for Z (the relationship matrix between path and deg), default 1.0
- beta_Z: the beta prior for Z (the relationship matrix between path and deg), default 1.0
- leaky: the OR-gate function should be leaky or not whether or not, default True
- max_iter: the maximum number of iteration for optimization, default 100

**Note: Need to remove samples with no mutations!**

```
def synthetic_mut2deg(n_path=5, n_sample=1000, n_deg=1000, n_mutation=1000, p_mutation=0.01, p_m2p=(0.02,0.05), p_p2d=(0.05,0.1)):
    mut = np.random.binomial(1, p_mutation, (n_sample, n_mutation))
    U = np.zeros((n_mutation, n_path))
    Z = np.zeros((n_path, n_deg))
    prior_U = np.random.uniform(p_m2p[0], p_m2p[1], n_path)
    prior_Z = np.random.uniform(p_p2d[0], p_p2d[1], n_path)
    for i in range(n_path):
        U[:,i] = np.random.binomial(1, prior_U[i], n_mutation)
        Z[i,:] = np.random.binomial(1, prior_Z[i], n_deg)
    path = mut.dot(U)
    mut1 = np.zeros(mut.shape)
    path[path>1] = 1
    for i in range(path.shape[1]):
        present_sample = np.where(path[:,i]==1)[0] # Index of samples where the ith pathways was perturbed
        for j in present_sample:
            exclusion = np.where(U[:,np.where(path[j]==0)[0]])[0] # The index of driver mutation that should not be chosen
            seed_mut = set(np.where(U[:,i])[0]).difference(set(exclusion))
            mutation = np.random.choice(list(seed_mut)) # Index of the chosen mutations that dysregulate the pathways
            mut1[j, mutation] = 1
            #mut1[:,U.sum(1)==0] = mut[:,U.sum(1)==0] # Keep the mutation status of passenger mutations
    path1 = mut1.dot(U)
    path1[path1>1] = 1

    deg = path.dot(Z)
    deg[deg>1] = 1

    return mut, U, path, Z, deg

def jaccard_quality(X_hat, Z_hat, X_true, Z_true):
    X_and = X_hat.T.dot(X_true)
    Z_and = Z_hat.dot(Z_true.T)
    
    X_or = np.zeros((X_hat.shape[1], X_hat.shape[1]))
    for i in range(X_hat.shape[1]):
        temp = X_hat.T[i] + X_true.T
        temp[temp > 1] = 1
        X_or[i] = temp.sum(1)
    X_qual = X_and / X_or

    Z_or = np.zeros((Z_hat.shape[0], Z_hat.shape[0]))
    for i in range(Z_hat.shape[0]):
        temp = Z_hat[i] + Z_true
        temp[temp > 1] = 1 
        Z_or[i] = temp.sum(1)
    Z_qual = Z_and/Z_or

    return np.max(X_qual * Z_qual, 0)


# Simulation experiment repeated 10 times
res_mut2deg = []
cos_mut2deg = []
for _ in range(10):
    mut, U, path, Z, deg = synthetic_mut2deg(n_path=5)
    
    model1 = mut2deg(np.float64(deg), np.float64(mut), pathway_size=5)
    path_pred1 = reconstruct(np.float64(mut>0.5), model1[0])
    res_mut2deg.append(np.mean(np.abs(reconstruct(path_pred1, model1[1])- deg))) # Caching reconstruction error
    cos_mut2deg.append(jaccard_quality(model1[0][:-1], model1[1], U, Z).mean()) # Caching Jaccard score
```
