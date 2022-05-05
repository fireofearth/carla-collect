import torch
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import utility.torchu
import utility as util

class Phi():
    def __init__(self):
        # TODO
        self.x_past_world = None
        # x_past_car has shape (B, A, T_past, D)
        #   The history of vehicle positions from
        #   car's local coordinate system.
        self.x_past_car = None
        self.x_past_grid = None
    
    def local_2_grid(self, *args):
        pass

    def local_2_world(self, *args):
        pass


class Bijection():

    def __init__(self):
        # whisker_points has shape (n whiskers, n points per whisker, D=2)
        self.whisker_points = None

    def __extract_map_features(self, feature_map, x_tml_grid):
        """Extract features from feature map using bilinear interpolation
        of car points.

        Parameters
        ==========
        feature_map : torch.Tensor
            Has shape (B, C, H, W).
        x_tml_grid : torch.Tensor
            Has shape (B, K, A, D=2).

        Returns
        =======
        torch.Tensor
            Has shape (B, K, A, C).
        """
        # x_tml_grid = x_tml_grid.reshape(-1, D)
        # h_map has shape (B, C, K, A)
        h_map = F.grid_sample(feature_map, x_tml_grid)
        # h_map has shape (B, K, A, C)
        h_map = torch.permute(h_map, (0, 2, 3, 1))
        return h_map
    
    def __extract_h_whisker(self, feature_map, h_map):
        """Extract features from feature map using bilinear interpolation
        of whisker points.

        Parameters
        ==========
        feature_map : torch.Tensor
            Has shape (B, C, H, W).
        x_tml_grid : torch.Tensor
            Has shape (B, K, A, D=2).

        Returns
        =======
        torch.Tensor
            Has shape (B, (n whiskers)*(n points per whisker)*C).
        """
        B, _, _, _ = h_map.shape
        # whisker_points has shape (B, n whiskers, n points per whisker, 2)
        whisker_points = util.torchu.expand_and_repeat(self.whisker_points, 0, B)
        # h_whisker has shape (B, C, n whiskers, n points per whisker)
        h_whisker = F.grid_sample(feature_map, whisker_points)
        # h_whisker has shape (B, (n whiskers)*(n points per whisker)*C)
        h_whisker = torch.permute(h_whisker, (0, 2, 3, 1)).reshape(B, -1)
        return h_whisker

    def __step(self, z, t, x_history, phi):
        B, K, A, T, D = z.shape
        #   x_{t-1} state
        feature_map = None # TODO
        x_tm1 = x_history[-1]
        # x_tml_grid has shape (B, K, A, D)
        x_tml_grid = phi.local_2_grid(x_tm1)
        # x_tml_world has shape (B, K, A, D)
        x_tml_world = phi.local_2_world(x_tm1)
        # h_map has shape (B, K, A, C)
        h_map = self.__extract_h_map(feature_map, x_tml_grid)
        # h_whisker has shape (B, F)
        h_whisker = self.__extract_h_whisker(feature_map, B)
        # h_social has shape (B, K, A, 2*C)
        h_social = []
        if A > 1:
            for a in range(A):
                # _h_map has the same shape as h_map
                _h_map = torch.cat((h_map[:, :, :a], h_map[:, :, :a+1]), dim=2)
                _h_map = torch.sum(_h_map, 2)
                # _h_map has shape (B, K, 1, 2*C)
                _h_map = torch.cat((h_map[:, :, a], _h_map), dim=3)
                h_social.append(_h_map)
            h_social = torch.cat(h_social, dim=2)
        else:
            h_social = torch.cat((h_map, h_map), dim=3)
        
        # TODO: get social spatial differences
        #       by computing pairwise differences between
        pass

    def forward(self, z, phi):
        """
        Parameters
        ==========
        z : torch.Tensor
            z has shape (B, K, A, T, D) where
            B is batch
            K is number of samples
            A is number of agents
            T is prediction horizon
            D is number of dimensions
        """
        B, K, A, T, D = z.shape
        x_past_car = util.torchu.expand_and_repeat(
                phi.x_past_car[..., -2:, :], 1, K)
        x_0 = x_past_car[..., -1, :]
        x_m1 = x_past_car[..., -2, :]
        # x_history : each has shape (B, K, A, D)
        x_history = [x_m1, x_0]
        for t in range(T):
            self.__step(z, t, x_history, phi)


class BijectiveDistribution():

    def __init__(self):
        self.base = MultivariateNormal(
            torch.zeros(2, dtype=torch.float),
            covariance_matrix=torch.full((2,), 1., dtype=torch.float),
            precision_matrix=None, scale_tril=None, validate_args=None
        )

    def log_p_x(self, x, phi):
        """Compute log p(x| phi)
        """
        pass

    def log_p_z(self, z, phi):
        """Compute log p(z| phi)
        """
        pass

    def sample(self, phi):
        """Sample x ~ p(x| phi)
        """
        shape = ... # TODO
        Z = self.base.sample(shape)
