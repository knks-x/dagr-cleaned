import torch
from dagr.utils.logging import log_bboxes
from dagr.utils.buffers import DetectionBuffer, format_data, PixelErrorBuffer
from dagr.utils.args import FLAGS
import tqdm
import numpy as np
import os
import wandb

args = FLAGS()

def to_npy(detections):
    return [{k: v.cpu().numpy() for k, v in d.items()} for d in detections]

def format_detections(sequences, t, detections):
    detections = to_npy(detections)
    for i, det in enumerate(detections):
        det['sequence'] = sequences[i]
        det['t'] = t[i]
    return detections

        
def run_test_without_visualization(loader, model, dataset: str, log_every_n_batch=-1, name="", compile_detections=False,
                                no_eval=False):
    model.eval()

    if not no_eval:
        mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width,
                                  classes=loader.dataset.classes)
        pixel_error_calc = PixelErrorBuffer(device="cuda")


    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Testing {name}")):
        data = data.cuda(non_blocking=True)

        data = format_data(data)
        detections, targets = model(data.clone())

        #global mAP
        if not no_eval:
            mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])         
        
        #global pixel error
        pixel_error_calc.update(detections, targets)
        
        if i % 5 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    #pixel error
    pixel_error_metrics = pixel_error_calc.compute()

    #mAP
    data = None
    if not no_eval:
        data = mapcalc.compute()

    return data, pixel_error_metrics

def run_test_with_visualization(loader, model, dataset: str, log_every_n_batch=-1, name="", compile_detections=False,
                                no_eval=False):
    """compile_detections: gather and return all raw detections from the entire test run"""

    model.eval()

    if not no_eval:
        mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width,
                                  classes=loader.dataset.classes)
        pixel_error_calc = PixelErrorBuffer(device="cuda")

    per_image_map = {}

    counter = 0
    if compile_detections:
        compiled_detections = []

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Testing {name}")):
        data = data.cuda(non_blocking=True)
        data_for_visualization = data.clone()

        data = format_data(data)
        detections, targets = model(data.clone())

        if compile_detections:
            compiled_detections.extend(format_detections(data.sequence, data.t1, detections))

        if log_every_n_batch > 0 and counter % log_every_n_batch == 0:
            log_bboxes(data_for_visualization, targets=targets, detections=detections, bidx=4,
                       class_names=loader.dataset.classes, key="testing/evaluated_bboxes")

        #global mAP
        if not no_eval:
            mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])         
        
        #per image accuracy
        if not no_eval:
            #extract current seqeunce + frame id (to use it later in vis_script)
            directory, img_idx_to_track_idx, mask, frame_id, sequence_id = loader.dataset.rel_index(i, r_sequence=True)                                       
            key=f"{sequence_id}_{frame_id}"

            per_image_metric = DetectionBuffer(height=data.height[0], width=data.width[0], classes=loader.dataset.classes)
            per_image_metric.update(detections, targets, dataset, data.height[0], data.width[0])

            per_image_map[key] = per_image_metric.compute()

            #test 
            if i < 5:  
                print(f"[DEBUG] {key} -> mAP: {per_image_map[key]['mAP']:.4f}")
            if i== 4:
                debug_path= "/data/sbaumann/nature-gnn/dagr/logs/dsec/detection/comparisons/per_image_map_debug.npy"
                np.save(debug_path,per_image_map, allow_pickle=True)

        #compute per pixel error
        pixel_error_calc.update(detections, targets)
        
        if i % 5 == 0:
            torch.cuda.empty_cache()

        counter += 1

    #save per image mAP
    if not no_eval:
        #save per image accuracy for visualization step   
        path = "/data/sbaumann/nature-gnn/dagr/logs/dsec/detection/comparisons"
        save_path = os.path.join(path, f"{args.exp_name}_per_image_map.npy")
        np.save(save_path, per_image_map, allow_pickle=True)
        print(f"Saved per-image mAP to {save_path}")

    #pixel error
    pixel_error_metrics = pixel_error_calc.compute()
    
    #mAP
    data = None
    if not no_eval:
        data = mapcalc.compute()
    
    torch.cuda.empty_cache()

    return (data, compiled_detections) if compile_detections else data, pixel_error_metrics
 
def run_test_pe(loader, model, dataset: str, log_every_n_batch=-1, name="", no_eval=False):
    model.eval()

    if not no_eval:
        pixel_error_calc = PixelErrorBuffer(device="cuda")

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Testing {name}")):
        data = data.cuda(non_blocking=True)

        data = format_data(data)
        detections, targets = model(data.clone())     
        
        #compute per pixel error
        pixel_error_calc.update(detections, targets)
        
        if i % 5 == 0:
            torch.cuda.empty_cache()

    #pixel error
    pixel_error_metrics = pixel_error_calc.compute()

    torch.cuda.empty_cache()

    return pixel_error_metrics