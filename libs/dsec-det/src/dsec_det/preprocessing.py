import numpy as np


def compute_img_idx_to_track_idx(t_track, t_image):                     #t_track=detection timestamp, t_image=frame timestamp. assign detection timestamp to matching frame timestamp
    x, counts = np.unique(t_track, return_counts=True)                  #save unique timestamps x=[t1,t,2,..] + count them  counts=[2, 3, ..]
    i, j = (x.reshape((-1,1)) == t_image.reshape((1,-1))).nonzero()     #reshape makes column/row out of input -> compare all values of x with all values of t_image ->save indices of matching in i(t_track) and j(t_image): x[i]=t_image[j
    deltas = np.zeros_like(t_image)                                     # array with zeros of shape t_image

    deltas[j] = counts[i]                                               #delta[j]: assign num of detections to indices saved in t_image ("at this timestamp there are n(count value) detections in t_track") 

    idx = np.concatenate([np.array([0]), deltas]).cumsum()
    return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")      #group detections per frame: e.g. first frame (slice)-> 2detections.


def compute_index(detection_index, discrete_timestamps):            
#added this function because of missing "extract_image_by_index"; see: https://github.com/uzh-rpg/dagr/issues/1
    mapping = np.zeros_like(discrete_timestamps, dtype="uint64")
    for i, t in enumerate(discrete_timestamps):
        mapping[i] = np.searchsorted(detection_index, t, side='right') - 1 
        mapping[i] = max(mapping[i], 0)
        mapping[i] = min(mapping[i], len(detection_index) - 1) 
    return mapping

    """compute index problem when vis_timestamp starts before image_timestamp. Wrongly matches first and last becasue of side and logic in the function
    So yes, the first vis timestamp gets “snapped” to the first image even though the second vis timestamp would be closer in absolute terms. This is why it looks like it’s jumping between first and last."""
    