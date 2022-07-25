import numpy as np
from src.symplectic import Symplectic, SymplecticFock, SymplecticXXPP
from src.gbs_matrices import GaussianMatrix, GBSMatrix
import thewalrus
import strawberryfields as sf


s = [1.25, 1.5, 1.75, 0.75]
M = len(s)

L = Symplectic.Lmat(M)

symp_fock = SymplecticFock.single_mode_squeezing(s)
symp_xxpp = SymplecticXXPP.single_mode_squeezing(s)


print(np.allclose(symp_xxpp, Symplectic.matrix_fock_to_xxpp(symp_fock) ))
print(np.allclose(symp_fock, Symplectic.matrix_xxpp_to_fock(symp_xxpp) ))

cov_fock = symp_fock @ GaussianMatrix.vacuum(M) @ symp_fock.T.conjugate()
cov_xxpp = symp_xxpp @ GaussianMatrix.vacuum(M) @ symp_xxpp.T.conjugate()

print(np.allclose(cov_xxpp, Symplectic.matrix_fock_to_xxpp(cov_fock)))

print(GaussianMatrix.is_valid_cov_fock(cov_fock))

A = GBSMatrix.Amat(cov_fock)

A_walrus = thewalrus.quantum.Amat(cov_xxpp, hbar=1)