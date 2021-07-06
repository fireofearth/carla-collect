"""Modeling of other vehicles for midlevel controller."""

import logging
import collections

import numpy as np
import pandas as pd
import scipy.spatial
import control
import control.matlab
import docplex.mp
import docplex.mp.model
import carla

import utility as util
import carlautil
import carlautil.debug

class OVehicle(object):

    @classmethod
    def from_trajectron(cls, node, T, ground_truth, past,
            latent_pmf, predictions, filter_pmf=0.1, bbox=[4.5, 2.5]):
        """Create other vehicle from Trajectron++ prediction.

        Parameters
        ==========
        ground_truth : np.array
            Array of ground truth trajectories of shape (T_gt_v, 2)
            T_gt_v is variable
        past : np.array
            Array of past trajectories of shape (T_past_v, 2)
            T_past_v is variable
        latent_pmf : np.array
            Array of past trajectories of shape (latent_states)
            Default settings in Trajectron++ sets latent_states to 25
        predictions : list of np.array
            List of predictions indexed by latent value.
            Each set of prediction corresponding to a latent is size (number of preds, T_v, 2)
        filter_pmf : float
            Filter value to apply on PMF. All predictions p(y|x,z) with PMF p(z) >= filter_pmf
            are placed in cluster z. Thre rest of the predictions are sorted to the cluster
            closest to an existing cluster.
        bbox : list of float
            Vehicle bounding box of form [longitudinal axis, lateral axis] in meters.
        """
        n_states = len(predictions)
        n_predictions = sum([p.shape[0] for p in predictions])
        
        """Heejin's control code"""
        pos_last = past[-1]
        # Masking to get relevant predictions.
        latent_mask = np.argwhere(latent_pmf > filter_pmf).ravel()
        masked_n_predictions = 0
        masked_n_states = latent_mask.size
        masked_pred_positions = []
        masked_pred_yaws = []
        masked_init_center = np.zeros((masked_n_states, 2))

        for latent_idx, latent_val in enumerate(latent_mask):
            """Preprocess the predictions that correspond to the latents in the mask"""
            ps = predictions[latent_val]
            n_p = ps.shape[0]
            yaws = np.zeros((n_p, T))
            # diff = (ps[:,0,1] - pos_last[1])**2  - ( ps[:,0,0] - pos_last[0])**2
            # TODO: skipping the i_keep_yaw code
            yaws[:,0] = np.arctan2(ps[:,0,1] - pos_last[1], ps[:,0,0] - pos_last[0])
            for t in range(1, T):
                # diff = (ps[:,t,1] - ps[:,t-1,1])**2  - (ps[:,t,0] - ps[:,t-1,0])**2
                # TODO: skipping the i_keep_yaw code
                yaws[:,t] = np.arctan2(ps[:,t,1] - ps[:,t-1,1], ps[:,t,0] - ps[:,t-1,0])
            masked_pred_positions.append(ps)
            masked_pred_yaws.append(yaws)
            masked_n_predictions += n_p
            masked_init_center[latent_idx] = np.mean(ps[:, T-1], axis=0)
        latent_neg_mask = np.in1d(np.arange(n_states), latent_mask, invert=True)
        latent_neg_mask = np.arange(n_states)[latent_neg_mask]
        for latent_val in latent_neg_mask:
            """Group rare latent values to the closest common latent variable"""
            ps = predictions[latent_val]
            if ps.size == 0:
                continue
            n_p = ps.shape[0]
            yaws = np.zeros((n_p, T))
            # TODO: skipping the i_keep_yaw code
            yaws[:,0] = np.arctan2(ps[:,0,1] - pos_last[1], ps[:,0,0] - pos_last[0])
            for t in range(1, T):
                # TODO: skipping the i_keep_yaw code
                yaws[:,t] = np.arctan2(ps[:,t,1] - ps[:,t-1,1], ps[:,t,0] - ps[:,t-1,0])
            
            dist = scipy.spatial.distance_matrix(ps[:,T-1,:], masked_init_center)
            p_cluster_ids = np.argmin(dist, axis=1)
            for idx in range(masked_n_states):
                tmp_ps = ps[p_cluster_ids == idx]
                if tmp_ps.size == 0:
                    continue
                tmp_yaws = yaws[p_cluster_ids == idx]
                masked_pred_positions[idx] = np.concatenate(
                        (masked_pred_positions[idx], tmp_ps,))
                masked_pred_yaws[idx] = np.concatenate(
                        (masked_pred_yaws[idx], tmp_yaws,))
            masked_n_predictions += n_p

        masked_latent_pmf = np.zeros(latent_mask.shape)
        for idx in range(masked_n_states):
            """Recreate the PMF"""
            n_p = masked_pred_positions[idx].shape[0]
            masked_latent_pmf[idx] = n_p / float(masked_n_predictions)
        
        return cls(node, T, past, ground_truth, masked_latent_pmf,
                masked_pred_positions, masked_pred_yaws, masked_init_center,
                bbox)

    def __init__(self, node, T, past, ground_truth, latent_pmf,
            pred_positions, pred_yaws, init_center, bbox):
        self.node = node
        self.T = T
        self.past = past
        self.ground_truth = ground_truth
        self.latent_pmf = latent_pmf
        self.pred_positions = pred_positions
        self.pred_yaws = pred_yaws
        self.init_center = init_center
        self.bbox = bbox
        self.n_states = self.latent_pmf.size
        self.n_predictions = sum([p.shape[0] for p in self.pred_positions])
