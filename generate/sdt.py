def get_sdt(bevs, filename_keys, sdt_clip_thresh, stamp=True, sdt_zero_h=0, sdt_zero_w=0., sdt_params={}, sdt_params_name=''):
    """

    :param bevs: 
    :param filename_keys: 
    :param sdt_clip_thresh: 
    :param stamp: Whether to zero-out a region in the features corresponding to the car.
    :param sdt_params: params to create the SDT
    :param sdt_params_name: extra key part to specify SDT creation method.
    :returns: 
    :rtype: 

    """
    B, H, W, C = bevs.shape
    for b in range(B):
        # Get the filename of this datum, we'll use it as a key.
        datum_filename = filename_keys[b]
        sdt_filename = datum_filename + sdt_params_name + '_sdt.npz'
            
        if os.path.isfile(sdt_filename):
            try: _load_sdt(bevs, sdt_filename, b)
            except ValueError as e:
                log.error("Caught when trying to load: {}".format(e))
                # Remove it so we can recreate it.
                try: os.remove(sdt_filename)
                except: pass
                _create_sdt(
                    bevs=bevs,
                    sdt_filename=sdt_filename,
                    b=b,
                    H=H,
                    W=W,
                    C=C,
                    sdt_clip_thresh=sdt_clip_thresh,
                    stamp=stamp,
                    sdt_zero_h=sdt_zero_h,
                    sdt_zero_w=sdt_zero_w,
                    sdt_params=sdt_params)
        else:
            _create_sdt(
                bevs=bevs,
                sdt_filename=sdt_filename,
                b=b,
                H=H,
                W=W,
                C=C,
                sdt_clip_thresh=sdt_clip_thresh,
                stamp=stamp,
                sdt_zero_h=sdt_zero_h,
                sdt_zero_w=sdt_zero_w,
                sdt_params=sdt_params)
    # Return the output.
    return bevs

def _load_sdt(bevs, sdt_filename, b):
    # Load the SDT into the bevs.
    sdt_b = np.load(sdt_filename, allow_pickle=True)['arr_0']
    bevs[b] = sdt_b

def create_sdt(bevs, directory, filename, )

def _create_sdt(bevs, sdt_filename, b, H, W, C, sdt_clip_thresh, stamp=True, sdt_zero_h=0, sdt_zero_w=0., sdt_params={}, save=True):
    # Create the SDT for this batch item.
    sdt_b = bevs[b]
    if stamp:
        # Stamp zeros out around the ego-car.
        center = (H / 2, W / 2 + 1)
        sdt_b[int(np.floor(center[0] - sdt_zero_h / 2)):int(np.ceil(center[0] + sdt_zero_h / 2)),
              int(np.floor(center[1] - sdt_zero_w / 2)):int(np.ceil(center[1] + sdt_zero_w / 2))] = 0.
    for c in range(C):
        float_input_array = sdt_b[..., c]
        binary_input_array = float_input_array > sdt_clip_thresh
        sdt_b[..., c] = npu.signed_distance_transform(binary_input_array, **sdt_params)
    if save:
        # Don't ever overwrite anything. TODO may cause issues during simultaneous training.
        assert(not os.path.isfile(sdt_filename))
        # Save the SDT
        np.savez_compressed(sdt_filename, sdt_b)