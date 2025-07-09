import torch
from scipy.spatial.transform import Rotation

def covariance_to_quaternion_and_scale(covariance, device='cpu'):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        # U, S, V = torch.linalg.svd(covariance)
        S, U = torch.linalg.eig(covariance)
        S = S.real
        U = U.real
        print(f"S: \n{S}, U: \n{U}")
        # Take the real part of S and U
        rotation = U
        identity_check = torch.allclose(
        rotation.transpose(-1, -2) @ rotation, 
        torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand_as(rotation),
        atol=1e-6
        )
        determinant_check = torch.allclose(
            torch.linalg.det(rotation), 
            torch.ones(rotation.shape[:-2], device=rotation.device, dtype=rotation.dtype),
            atol=1e-6
        )
        print(f"The rotation matrix is not orthonormal, identity_check: {identity_check}, determinant_check: {determinant_check}, det: {torch.linalg.det(rotation)}.")
        # if not (identity_check and determinant_check):
        #     raise ValueError(f"The rotation matrix is not orthonormal, identity_check: {identity_check}, determinant_check: {determinant_check}, det: {torch.linalg.det(rotation)}.")
        # else:
        #     print("The rotation matrix is orthonormal.")

        # Swap eigenvalue positions and adjust U to ensure determinant of 1
        negative_determinants = torch.linalg.det(U) < 0
        U[negative_determinants,..., -1] = -U[negative_determinants,..., -1]
        # print(F"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)

        # The rotation matrix is U*Vt
        # rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix = U
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # print(f"covariance: {covariance[0,...]}")
        # print(f"RSStRt: {torch.bmm(torch.bmm(rotation_matrix,scale.diag_embed()),scale.diag_embed().transpose(-2, -1))}")

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()
        quaternion = torch.from_numpy(quaternion).to(device)

        return quaternion, scale