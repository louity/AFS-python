import afs
import argparse
import numpy as np
import os
from sklearn import linear_model, model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser('Compute AFS descriptor')
parser.add_argument('--database', choices=['EAM', 'MEAM'],  help="database chosen", required=True)
parser.add_argument('--r_cut', type=float,  help="cutoff radius", required=True)
parser.add_argument('--n_max', type=int,  help="n radial functions", required=True)
parser.add_argument('--l_max', type=int,  help="n angular functions", required=True)
args = parser.parse_args()


n_max, l_max, r_cut = args.n_max, args.l_max, args.r_cut

output_file = f'AFS_{args.database}_n={n_max}_l={l_max}_r={r_cut}.npz'
if os.path.isfile(output_file):
    print(f'reading descriptor file {output_file}...')
    descs = np.load(output_file)
    I2_AFS, I3_AFS, I4_AFS, V4_AFS, bulk_AFS = descs['I2'], descs['I3'], descs['I4'], descs['V4'], descs['bulk']
else:
    configs = np.load(f'./configs_{args.database}.npz')
    bulk_config = np.load('config_bulk.npy')
    I2_configs, I3_configs, I4_configs, V4_configs = configs[f'I2_{args.database}'],  configs[f'I3_{args.database}'], configs[f'I4_{args.database}'], configs[f'V4_{args.database}']
    lattice = configs['lattice']
    dimensions = np.linalg.norm(lattice, axis=1)
    print('computing AFS desc for bulk, 1024 atoms...')
    bulk_AFS =  afs.compute_AFS_descriptors(bulk_config, n_max, l_max, r_cut, dimensions, reg_eigenvalues=1e-15).sum(axis=1)
    print('computing AFS desc for I2 (2 intersticials, 1026 atoms)...')
    I2_AFS =  afs.compute_AFS_descriptors(I2_configs, n_max, l_max, r_cut, dimensions, reg_eigenvalues=1e-15).sum(axis=1)
    print('computing AFS desc for I3 (3 intersticials, 1027 atoms)...')
    I3_AFS =  afs.compute_AFS_descriptors(I3_configs, n_max, l_max, r_cut, dimensions, reg_eigenvalues=1e-15).sum(axis=1)
    print('computing AFS desc for I4 (4 intersticials, 1028 atoms)...')
    I4_AFS =  afs.compute_AFS_descriptors(I4_configs, n_max, l_max, r_cut, dimensions, reg_eigenvalues=1e-15).sum(axis=1)
    print('computing AFS desc for V4 (4 vacancies, 1020 atoms)...')
    V4_AFS =  afs.compute_AFS_descriptors(V4_configs, n_max, l_max, r_cut, dimensions, reg_eigenvalues=1e-15).sum(axis=1)

    print(f'saving to file {output_file}')
    np.savez(output_file, I2=I2_AFS, I3=I3_AFS, I4=I4_AFS, V4=V4_AFS, bulk=bulk_AFS)

I2_AFS, I3_AFS, I4_AFS, V4_AFS = I2_AFS.reshape((-1, n_max*(l_max+1))), I3_AFS.reshape((-1, n_max*(l_max+1))), I4_AFS.reshape((-1, n_max*(l_max+1))), V4_AFS.reshape((-1, n_max*(l_max+1)))

entropy_file = f'entropies_{args.database}.npz'
entropies = np.load(entropy_file)
I2_entropies, I3_entropies, I4_entropies, V4_entropies = entropies['I2'], entropies['I3'], entropies['I4'], entropies['V4']

nonzero_I2, nonzero_I3, nonzero_I4, nonzero_V4 = np.nonzero(I2_entropies), np.nonzero(I3_entropies), np.nonzero(I4_entropies), np.nonzero(V4_entropies)

X = np.concatenate([I2_AFS[nonzero_I2], I3_AFS[nonzero_I3], I4_AFS[nonzero_I4], V4_AFS[nonzero_V4]], axis=0)
y = np.concatenate([I2_entropies[nonzero_I2], I3_entropies[nonzero_I3], I4_entropies[nonzero_I4], V4_entropies[nonzero_V4]], axis=0)

print(f'X {X.shape}, y {y.shape}')

clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False)

alpha_min, min_rmse = -1, np.inf
for alpha in 10.**(-np.arange(-2, 10)):
    clf = linear_model.Ridge(fit_intercept=False, alpha=alpha)

    n_crossval_folds = 5
    y_prediction = model_selection.cross_val_predict(clf, X=X, y=y, cv=n_crossval_folds)
    cv_mae = mean_absolute_error(y_prediction, y)
    cv_rmse = np.sqrt(mean_squared_error(y_prediction, y))
    if cv_rmse < min_rmse:
        min_rmse = cv_rmse
        alpha_min = alpha
    print(f'BayesianRidge {alpha} {n_crossval_folds} folds cross val mae {cv_mae:.4f} kB, rmse {cv_rmse:.4f} kB')

clf = linear_model.Ridge(fit_intercept=False, alpha=alpha_min)

clf.fit(X, y)
y_predict = clf.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, y_predict))
train_mae = mean_absolute_error(y, y_predict)
print(f'Ridge alpha={alpha_min} train mae {train_mae:.4f} kB, train rmse {train_rmse:.4f}kB')
