# Angular Fourier Series descriptors

To reproduce the results of [Machine learning surrogate models for prediction of point defect vibrational entropy](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.4.063802).
Python implementation of the chemical environment descriptor Angular Fourier Series defined in the paper
 [On representing chemical environments](https://doi.org/10.1103/PhysRevB.87.184115) by Bartok, Kondor, Csanyi.

## Dependencies
* python3 >= 3.6  (for f strings)
* numpy >= 1.15
* scipy >= 1.4
* scikit-learn >= 0.19

## Implementation details

  * Periodic boundary condition are handled via replication of the atomic positions.
  * Atom neighbors search are done using `scipy.spatial.cKDTree`.
  * Square root of the scalar product matrix ( formula (25) section III.D, page 7 in the paper 'On representing chemical environments') is computed via diagonalization using scipy.
  Regularization term is added on the diagonal to prevent from small negative eigenvalues (magnitude ~ 1e-16) due to numerical artifacts.
  * Computations are vectorized using numpy functions.

## Usage

To reproduce the results in figure 2 of the paper 'Machine learning surrogate ...' for EAM database, run
```
python main.py --database EAM --l_max 10 --n_max 20 --r_cut 5.
```
and for MEAM database, run
```
python main.py --database MEAM --l_max 10 --n_max 20 --r_cut 5.
```

If you use this work, please cite :
```
@article{lapointe2020machine,
  title={Machine learning surrogate models for prediction of point defect vibrational entropy},
  author={Lapointe, Clovis and Swinburne, Thomas D and Thiry, Louis and Mallat, St{\'e}phane and Proville, Laurent and Becquart, Charlotte S and Marinica, Mihai-Cosmin},
  journal={Physical Review Materials},
  volume={4},
  number={6},
  pages={063802},
  year={2020},
  publisher={APS}
}
```

