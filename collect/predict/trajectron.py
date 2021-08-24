import numpy as np
import torch

from model.dataset import get_timesteps_data
from model.model_utils import ModeKeys

def generate_vehicle_latents(eval_stg,
            scene, t, ph, num_samples=200,
            z_mode=False, gmm_mode=False,
            full_dist=False, all_z_sep=False):
    """Generate predicted trajectories and their corresponding
    latent variables.

    Returns
    =======
    scene : Scene
        The nuScenes scene
    t : int
        The timestep in the scene
    z : ndarray
        Has shape (number of vehicles, number of samples)
    zz : ndarray
        Has shape (number of vehicles, number of samples, number of latent values)
    predictions : ndarray
        Has shape (number of vehicles, number of samples, prediction horizon, D)
    nodes : list of Node
        Has size (number of vehicles)
        List of vehicle nodes
    predictions_dict : dict
        Contains map of predictions by timestep, by vehicle node
    """
    # Trajectron.predict() arguments
    timesteps = np.array([t])
    min_future_timesteps = 0
    min_history_timesteps = 1

    # In Trajectron.predict() scope
    node_type = eval_stg.env.NodeType.VEHICLE
    if node_type not in eval_stg.pred_state:
        raise Exception("fail")

    model = eval_stg.node_models_dict[node_type]

    # Get Input data for node type and given timesteps
    batch = get_timesteps_data(
            env=eval_stg.env, scene=scene,
            t=timesteps, node_type=node_type,
            state=eval_stg.state, pred_state=eval_stg.pred_state,
            edge_types=model.edge_types,
            min_ht=min_history_timesteps, max_ht=eval_stg.max_ht,
            min_ft=min_future_timesteps, max_ft=min_future_timesteps,
            hyperparams=eval_stg.hyperparams)
    
    # There are no nodes of type present for timestep
    if batch is None:
        raise Exception("fail")

    (first_history_index,
    x_t, y_t, x_st_t, y_st_t,
    neighbors_data_st,
    neighbors_edge_value,
    robot_traj_st_t,
    map), nodes, timesteps_o = batch

    x = x_t.to(eval_stg.device)
    x_st_t = x_st_t.to(eval_stg.device)
    if robot_traj_st_t is not None:
        robot_traj_st_t = robot_traj_st_t.to(eval_stg.device)
    if type(map) == torch.Tensor:
        map = map.to(eval_stg.device)

    # MultimodalGenerativeCVAE.predict() arguments
    inputs = x
    inputs_st = x_st_t
    first_history_indices = first_history_index
    neighbors = neighbors_data_st
    neighbors_edge_value = neighbors_edge_value
    robot = robot_traj_st_t
    prediction_horizon = ph

    # In MultimodalGenerativeCVAE.predict() scope
    mode = ModeKeys.PREDICT

    x, x_nr_t, _, y_r, _, n_s_t0 = model.obtain_encoded_tensors(mode=mode,
                                                            inputs=inputs,
                                                            inputs_st=inputs_st,
                                                            labels=None,
                                                            labels_st=None,
                                                            first_history_indices=first_history_indices,
                                                            neighbors=neighbors,
                                                            neighbors_edge_value=neighbors_edge_value,
                                                            robot=robot,
                                                            map=map)

    model.latent.p_dist = model.p_z_x(mode, x)
    latent_probs = model.latent.get_p_dist_probs() \
            .cpu().detach().numpy()
    latent_probs = np.squeeze(latent_probs)

    z, num_samples, num_components = model.latent.sample_p(num_samples,
                                                        mode,
                                                        most_likely_z=z_mode,
                                                        full_dist=full_dist,
                                                        all_z_sep=all_z_sep)
    
    _, predictions = model.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)

    z = z.cpu().detach().numpy()
    zz = z
    # z has shape (number of samples, number of vehicles, number of latent values)
    # z[i,j] gives the latent for sample i of vehicle j

    # Back to Trajectron.predict() scope    
    predictions = predictions.cpu().detach().numpy()
    # predictions has shape (number of samples, number of vehicles, prediction horizon, D)

    predictions_dict = dict()
    for i, ts in enumerate(timesteps_o):
        if ts not in predictions_dict.keys():
            predictions_dict[ts] = dict()
        predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))
    
    z = np.swapaxes(np.argmax(z, axis=-1), 0, 1)
    predictions = np.swapaxes(predictions, 0, 1)
        
    return z, zz, predictions, nodes, predictions_dict, latent_probs