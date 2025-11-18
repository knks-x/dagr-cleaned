import cv2
import wandb
import numpy as np
import os
from pathlib import Path

from dsec_det.directory import DSECDirectory
from dsec_det.io import extract_from_h5_by_timewindow, extract_image_by_index, load_start_and_end_time
from dsec_det.preprocessing import compute_index

from dagr.visualization.bbox_viz import draw_bbox_on_img, draw_gtbox_on_img
from dagr.visualization.event_viz import draw_events_on_image

from dagr.utils.logging import set_up_logging_directory
from dagr.utils.args import FLAGS


def load_detections(folder, detection_type, vis_timestamps):
    detections_file = folder / f"{detection_type}_detections.npy"
    if not detections_file.exists():
        raise FileNotFoundError(f"Could not find {detections_file}")
    detections = np.load(detections_file)
    detection_timestamps = np.unique(detections['t'])
    step_index_to_boxes_index = compute_index(detection_timestamps, vis_timestamps) #array with relevant indices: for the timestamp that should be visualized we need image with this corresponding index
    
    return detections, detection_timestamps, step_index_to_boxes_index


def load_gt_tracks(test_root, sequence, vis_timestamps, filter_classes=False):
    tracks_path = Path(test_root) / sequence / "object_detections" / "left" / "tracks.npy"
    if not tracks_path.exists():
        raise FileNotFoundError(f"Ground truth tracks not found: {tracks_path}")
    tracks=np.load(tracks_path)
    if filter_classes:
        tracks = filter_gt_classes(tracks, valid_class_ids=(0, 2))                  #filter before saving the timestamps and finding the matching track for current vis_timestamp
    tracks_timestamps = np.unique(tracks['t'])
    step_index_to_boxes_index = compute_index(tracks_timestamps, vis_timestamps)

    return tracks, tracks_timestamps, step_index_to_boxes_index

def filter_gt_classes(gt_tracks, valid_class_ids=(0, 2)):                               
    "Filter ground truth tracks to only include classes present in detection model. In this Case 0 and 2 (pedestrians + cars)"

    mask = np.isin(gt_tracks['class_id'], valid_class_ids)
    filtered_gt = gt_tracks[mask]
    return filtered_gt


def draw_variant(det, ts, idx, step, img, scale=1):                                    #det=detections, ts=timestamps, idx="step_index_to_image_index":array with relevant images (that matches the visualization timestamp the best)
    boxes_index = idx[step]                                #which boxes do we need? idx[step] returns index in detection array (det) (which corresponds to the relevant image/boxes)
    boxes_timestamp = ts[boxes_index]                      #get timestamp of that image/boxes
    boxes = det[det['t'] == boxes_timestamp]               #get all the boxes for that timestamp. boxes= sub array of detections corresponding to the current image/time step (still has all the information that det has, such as conf, w,h,x,y,...)
   
    img_copy = img.copy()
    img_copy, filtered_boxes = draw_bbox_on_img(                                                            
        img_copy,
        scale * boxes['x'], scale * boxes['y'], scale * boxes['w'], scale * boxes["h"],
        boxes["class_id"], boxes['class_confidence'], conf=0.3, nms=0.65, filtered_boxes=True
        )
    return img_copy, filtered_boxes


def draw_gt_boxes(gt, ts, idx, step, img, scale=1):
    boxes_index = idx[step]
    boxes_timestamp = ts[boxes_index]
    boxes = gt[gt['t'] == boxes_timestamp]

    img_copy = img.copy()
    img_copy = draw_gtbox_on_img(
        img_copy,
        scale * boxes['x'], scale * boxes['y'], scale * boxes['w'], scale * boxes["h"],
        boxes["class_id"]
        )
    return img_copy, boxes

debug=True
if debug:
    def debug_print_boxes(name, arr):
        print(f"\n--- {name} ---")
        print("Type:", type(arr))
        print("Shape:", arr.shape)
        print("Dtype:", arr.dtype)
        if arr.shape[0] > 0:
            print("First entry:", arr[0])
            print("Field names:", arr.dtype.names)
        else:
            print("Array is empty")

if __name__ == '__main__':
    args = FLAGS()

    assert args.dataset_directory_vis.exists()

    seq_folder = Path(args.detections_folder) / args.sequence

    dsec_directory = DSECDirectory(args.dataset_directory_vis / args.sequence)                      #here is the test data
    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)


    t0, t1 = load_start_and_end_time(dsec_directory)
    vis_timestamps = np.arange(t0, t1, step=args.vis_time_step_us)
    step_index_to_image_index = compute_index(dsec_directory.images.timestamps, vis_timestamps) #array with indices of closest actual detection timestamp: image/detection timestamps(timestamps at which detections were produced) + timestamps of visualization(e.g. every 1ms)
                                                                                                #for each visualization step: get the index of the detection timestamp closest to it.
                                                                                                #save the index of detection timestamp in the new array. length of array = len of vis_timestamp
    
    #load pred detections and gt tracks
    front_det, front_ts, front_idx = load_detections(seq_folder, "front", vis_timestamps)               #front_det= all detecions of that sequence, front_ts= alle unique timestamps, front_idx= array of all matching indices via step_index_to_boxes_index
    back_det, back_ts, back_idx = load_detections(seq_folder, "back", vis_timestamps)
    gt_tracks, gt_ts, gt_idx = load_gt_tracks(args.dataset_directory_vis, args.sequence, vis_timestamps, filter_classes=True) 
    print(len(front_det), len(back_det), len(gt_tracks))
    debug_print_boxes("GT_Tracks", gt_tracks)
    #debug class_id
    debug_classid = False
    if debug_classid:
        front_unique = np.unique(front_det['class_id']) if len(front_det) > 0 else []
        back_unique = np.unique(back_det['class_id']) if len(back_det) > 0 else []
        gt_unique = np.unique(gt_tracks['class_id']) if len(gt_tracks) > 0 else []

        print("=== Class ID Debug ===")
        print(f"Front unique class IDs: {front_unique}")
        print(f"Back unique class IDs:  {back_unique}")
        print(f"GT unique class IDs:    {gt_unique}")
        print("=======================")
    
    scale = 2

    #load the numpy with the per_image_map 
    per_image_map_path_front="logs/dsec/detection/comparisons/test_front_det_per_image_map.npy"
    per_image_map_path_back="logs/dsec/detection/comparisons/test_back_det_per_image_map.npy"
    per_image_map_front = np.load(per_image_map_path_front, allow_pickle=True).item()
    per_image_map_back = np.load(per_image_map_path_back, allow_pickle=True).item()

    #debug per image mAP:
    debug_image_map=False
    if debug_image_map:
        print("=== Debugging per_image_map contents ===")

        # Keys info
        print(f"Front map: {len(per_image_map_front)} entries, "
            f"min key={min(per_image_map_front.keys())}, "
            f"max key={max(per_image_map_front.keys())}")

        # Sample a few keys + values
        front_sample_keys = list(per_image_map_front.keys())[:10]
        back_sample_keys = list(per_image_map_back.keys())[:10]

        print("\nFront sample keys:", front_sample_keys)
        for k in front_sample_keys:
            print(f"  key={k}, value type={type(per_image_map_front[k])}, value={per_image_map_front[k]}")
    
    debug_single_sequence=False
    if debug_single_sequence:
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("interlaken_00_a")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("interlaken_00_b")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("interlaken_01_a")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("thun_01_a")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("thun_01_b")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("thun_02_a")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("zurich_city_12_a")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("zurich_city_13_a")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("zurich_city_13_b")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("zurich_city_14_a")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("zurich_city_14_b")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("zurich_city_14_c")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")
        interlaken_keys = [k for k in per_image_map_front.keys() if k.startswith("zurich_city_15_a")]
        print(f"Number of Interlaken_00_a entries: {len(interlaken_keys)}")


    for step, t in enumerate(vis_timestamps):                                                                       #step is the index loop of visualization timestamps(1,2,3,..).
                                                                                                                    #t is actual timestamp of vis_timestamps(0,1000,2000,3000,..) - not relative to start  - absolute timestamps.
        
        #load base image
        print(f"In visualization time window: Visualization step {step} - Visualization timestamp {t/1000:.2f}ms")
        #TODO: realign first index because of mismatch of compute index logic
        image_index = step_index_to_image_index[step]                                                               #index of the sequence image whose timestamp is closest to t.- e.g. step_index_to_image_index[1]= 2. at step=1/vis_timestep=1ms we need image/frame frome sequence with index 2
        base_img = extract_image_by_index(dsec_directory.images.image_files_distorted, image_index)

        #events on image
        events = extract_from_h5_by_timewindow(dsec_directory.events.event_file, t - args.event_time_window_us, t)
        base_img = draw_events_on_image(base_img, events['x'], events['y'], events['p'])

        #debug
        debug_ts=False
        if debug_ts:
            print("--1--")
            print("GT timestamps (gt_ts) shape:", gt_ts.shape)
            print("First 5 GT timestamps:", gt_ts[:5])
            print("Last 5 GT timestamps:", gt_ts[-5:])

            print("--2--")
            vis_t = vis_timestamps[step]
            gt_index_for_step = gt_idx[step]
            print(f"Visualization timestamp: {vis_t}")
            print(f"Mapped GT timestamp: {gt_ts[gt_index_for_step]}")
            print(f"GT index for step: {gt_index_for_step}")
            print(f"Image Index for step: {image_index}")
            
            """print("--3--")
            boxes_for_step = gt_tracks[gt_tracks['t'] == gt_ts[gt_index_for_step]]
            print(f"Number of GT boxes at mapped timestamp: {len(boxes_for_step)}")
            if len(boxes_for_step) > 0:
                print("Sample GT box:", boxes_for_step[0])"""

        #gt boxes on base image
        gt_img, gt_boxes = draw_gt_boxes(gt_tracks, gt_ts, gt_idx, step, base_img, scale=1)     #don't scale gt_values!

        #pred boxes on base image
        front_img, front_boxes = draw_variant(front_det, front_ts, front_idx, step, gt_img, scale)
        back_img, back_boxes = draw_variant(back_det, back_ts, back_idx, step, gt_img, scale)
        
        combined = np.concatenate([gt_img, front_img, back_img], axis=1)

        #compute length of sequence
        #seq_duration_us = vis_timestamps[-1] - vis_timestamps[0]
        #seq_duration_ms = seq_duration_us / 1_000
        #print(f"Duration of sequence: {seq_duration_ms}ms")


        num_front = len(front_boxes)                         #len(boxes) is literally the number of detection boxes for that timestamp - here only filtered boxes >0.3 conf
        num_back = len(back_boxes)
        num_gt = len(gt_boxes)

        #debug
        debug_boxes=False
        if debug_boxes:
            "Length of boxes after Filtering"
            print(f"front -> num_boxes={num_front}")
            print(f"back -> num_boxes={num_back}")
            print(f"gt -> num_boxes={num_gt}")
        #get metrics
        print("Current image index:", image_index)          #first image_idx weirdly high- no correlation with front
        print("Current sequence:", args.sequence)

        key = f"{args.sequence}_{image_index}"
        if key in per_image_map_front:
            metrics_front = per_image_map_front[key]
        else:
            print(f"Warning: {key} not found in per_image_map_front")
            key = f"{args.sequence}_{0}"
            metrics_front = per_image_map_front[key]

        key = f"{args.sequence}_{image_index}"
        if key in per_image_map_back:
            metrics_back = per_image_map_back[key]
        else:
            print(f"Warning: {key} not found in per_image_map_back")
            key = f"{args.sequence}_{0}"
            metrics_back = per_image_map_back[key]

        #debug
        debug=True
        if debug:
            #for name, boxes in zip(["front", "back", "gt"], [front_boxes, back_boxes, gt_boxes]):
                #print(f"{name} -> num_boxes={len(boxes)}")
            print(f"Front mAP for Index {image_index} -> {metrics_front['mAP']}")
            print(f"Back mAP for Index {image_index} -> {metrics_back['mAP']}")

        logic_boxes=False
        if logic_boxes:
            for name, boxes in zip(["front", "back", "gt"], [front_boxes, back_boxes, gt_boxes]):
                print(f"{name} -> num_boxes={len(boxes)}")
                if num_front != num_back or num_front != num_gt or num_gt != num_back:
                    wandb.log({ 
                        f"{args.sequence}_Image{image_index}": wandb.Image( 
                            combined, caption=f"Visualization step {step} - Visualization timestamp {t/1000:.2f}ms - mAP front: {metrics_front:.2f} - mAP back: {metrics_back:.2f}"
                            )
                    })
                    print(f"Logged Image {image_index}")

        logic_map=False
        if logic_map:
                print(f"Front mAP for Index {image_index} -> {metrics_front['mAP']}")
                print(f"Back mAP for Index {image_index} -> {metrics_back['mAP']}")
                if metrics_front['mAP']  < 0.3 or metrics_back['mAP'] < 0.3:                 #or use this for compare amount of boxes: num_front != num_back or num_front != num_gt or num_gt != num_back:                      
                    wandb.log({ 
                        f"{args.sequence}_Image{image_index}": wandb.Image( 
                            combined, caption=f"Visualization step {step} - Visualization timestamp {t/1000:.2f}ms - mAP front: {metrics_front['mAP']:.2f} - mAP back: {mmetrics_back['mAP']:.2f}"
                            )
                    })
                    print(f"Logged Image {image_index}")
                    break  # log once per step if any is odd

        if args.write_to_output:
            out_path = Path(output_directory) / args.sequence
            out_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path / f"{step:06d}.png"), combined)
