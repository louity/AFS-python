"""Python implementation of the Angular Fourier Series descriptors defined in the paper
'On representing chemical environments', DOI: https://doi.org/10.1103/PhysRevB.87.184115
"""
import numpy as np
import scipy
import scipy.spatial as spatial
try:
    from tqdm import tqdm
except ImportError: tqdm = lambda x: x

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def compute_W_matrix(n_max, reg=0.):
    """ W matrix involved in formula (25) section III.D (page 7) concerning
    orthonomalization of the radial functions \phi.

    W is defined as the square root of the scalar product matrix S(i,j) = <phi_i, phi_j>.
    Beacause phi functions are correlated, large n_max may cause negative
    eigenvalues that yield nans in sqrt matrix, so we added a reg term on the diagonal.

    Note that scipy.linalg.sqrtm can also be used, but similarly, complex value appear for large
    n_max.
    """
    overlapp_matrix = np.zeros((n_max, n_max))
    for alpha in range(1, n_max+1):
        for beta in range(1, n_max+1):
            overlapp_matrix[alpha-1, beta-1] = np.sqrt((5 + 2*alpha)*(5 + 2*beta)) /  (5 + alpha + beta)
            if alpha == beta and reg > 0:
                overlapp_matrix[alpha-1, beta-1] += reg


    eigvals, eigvecs = scipy.linalg.eigh(overlapp_matrix)

    if (eigvals < 0).sum():
        print('Negative eigenvalues in matrix W: {}'.format(eigvals))

    W_matrix = np.dot(np.dot(eigvecs, np.diag(1. / np.sqrt(eigvals))),  eigvecs.transpose())

    return W_matrix


def periodize_configuration(configuration, r_cut, dimensions):
    """Periodically replicate the atoms in a rectangular box that are distance <= r_cut of the faces.

    Parameters
        configuration: np.array of shape (n_atoms, 3)
            coordinates of the atoms to be periodized

        r_cut: float
            cutoff radius

        dimensions: np.array of shape (3,) (or list length 3)
            dimensions of the periodic rectangle

    Returns
        periodized_configuration: np.array of shape (n_atoms_periodized, 3)

        initial_atom_ids: np.array of shape (n_atoms_periodized, )
            ids of the periodized atoms in the initial configuration
    """
    periodized_configuration = []
    initial_atom_ids = []

    x_translation = np.array([[dimensions[0], 0, 0]], dtype=configuration.dtype)
    y_translation = np.array([[0, dimensions[1], 0]], dtype=configuration.dtype)
    z_translation = np.array([[0, 0, dimensions[2]]], dtype=configuration.dtype)

    mask_true = np.ones(configuration.shape[0], dtype=bool)

    for i_x, mask_x in [(-1., configuration[:, 0] > (dimensions[0] - r_cut)), (0., mask_true), (1., configuration[:, 0] < r_cut)]:
        for i_y, mask_y in [(-1., configuration[:, 1] > (dimensions[1] - r_cut)), (0., mask_true), (1., configuration[:, 1] < r_cut)]:
            for i_z, mask_z in [(-1., configuration[:, 2] > (dimensions[2] - r_cut)), (0., mask_true), (1., configuration[:, 2] < r_cut)]:
                mask = mask_x * mask_y * mask_z
                initial_atom_ids.append(np.nonzero(mask)[0])
                periodized_configuration.append(configuration[mask] + i_x*x_translation + i_y*y_translation + i_z*z_translation)

    periodized_configuration = np.concatenate(periodized_configuration, axis=0)
    initial_atom_ids = np.concatenate(initial_atom_ids, axis=0)

    return periodized_configuration, initial_atom_ids


def compute_AFS_descriptors(configurations, n_max, l_max, r_cut, dimensions,
                            radial_function_type='g_function', reg_eigenvalues=0.,
                            neighbors_in_r_cut=False, radial_tensor_product=False):
    """Implementation of the formula given in section III.G (page 9).
       The indices i and i' in the sum are interpreted as the neighbor indices of a central atom.
       If neighbors_in_r_cut=True, we add the constraint that the neighbors i and i' must be at a distance <= r_cut
    """
    assert radial_function_type in ['g_function', 'gaussian'], f'invalid radial function type {radial_function_type}'

    l_values = np.arange(l_max+1).reshape(1, 1, -1)

    if radial_function_type == 'g_function':
        W_matrix = compute_W_matrix(n_max, reg=reg_eigenvalues)
        alphas = np.arange(1, n_max+1).astype('float64').reshape((1, -1))
        exponents = alphas + 2
        normalizing_constants = np.sqrt(2*alphas+5) / np.power(r_cut, alphas+2.5)
    elif radial_function_type == 'gaussian':
        centers = np.linspace(0, r_cut, n_max, endpoint=False).reshape((1, -1))
        sigma = 0.5 * centers[0, 1]

    if radial_tensor_product:
        AFS_descriptors = np.zeros((configurations.shape[0], configurations.shape[1], n_max**2, l_max+1))
    else:
        AFS_descriptors = np.zeros((configurations.shape[0], configurations.shape[1], n_max, l_max+1))

    for i_config in tqdm(range(configurations.shape[0])):
        configuration = configurations[i_config]
        periodized_configuration, initial_atom_ids = periodize_configuration(configuration, r_cut, dimensions)
        point_tree = spatial.cKDTree(periodized_configuration)

        for i_atom in range(configuration.shape[0]):
            atom = configuration[i_atom:i_atom+1]
            neighbors_indices = [n_id for n_id in point_tree.query_ball_point(configuration[i_atom], r_cut) if initial_atom_ids[n_id] != i_atom]
            neighbors = periodized_configuration[neighbors_indices]
            r_vectors = neighbors - atom
            r_norms = np.linalg.norm(r_vectors, axis=1, keepdims=True)
            if radial_function_type == 'g_function':
                phi_functions = normalizing_constants * (r_cut - r_norms)**exponents
                radial_functions = np.dot(phi_functions, W_matrix)
            elif radial_function_type == 'gaussian':
                radial_functions = gaussian(r_norms, centers, sigma)

            r_normalized = r_vectors / r_norms
            cos_angles = np.dot(r_normalized, r_normalized.transpose())

            # triangular-upper mask corresponding to pairs (i,j) with i<j, i.e. pair of different atoms
            n_neighbors = neighbors.shape[0]
            triu_mask = np.arange(n_neighbors)[:,None] < np.arange(n_neighbors)
            if neighbors_in_r_cut:
                neighbors_indices_pdist_matrix = spatial.distance.squareform(spatial.distance.pdist(neighbors))
                neighbors_in_r_cut_mask = neighbors_indices_pdist_matrix < r_cut
                triu_mask *= neighbors_in_r_cut_mask

            # triangular-upper indices correspond to pairs (i,j) with i<j, i.e. pair of different atoms
            cos_angles = np.clip(cos_angles, -1, 1)
            angles_triu = np.arccos(cos_angles[triu_mask]).reshape(-1, 1, 1)

            cos_l_angles_triu = np.cos(l_values * angles_triu)

            if radial_tensor_product:
                radial_functions_product_triu = np.tensordot(radial_functions.transpose(), radial_functions, axes=0).transpose(1, 2, 0, 3)[triu_mask, :, :].reshape(-1, n_max**2, 1)
            else:
                radial_functions_product_triu = np.stack([np.dot(radial_functions[:, n:n+1], radial_functions[:, n:n+1].transpose())[triu_mask] for n in range(n_max)], axis=-1)[:, :, np.newaxis]

            AFS_descriptors[i_config, i_atom] += (radial_functions_product_triu * cos_l_angles_triu).sum(axis=0)

    return AFS_descriptors
