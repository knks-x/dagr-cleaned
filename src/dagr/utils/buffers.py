import numpy as np
import torch

from typing import List, Dict
from pathlib import Path

from .coco_eval import evaluate_detection
from yolox.utils import bboxes_iou


def diag_filter(bbox, height: int, width: int, min_box_diagonal: int = 30, min_box_side: int = 20):
    bbox[..., 0::2] = torch.clamp(bbox[..., 0::2], 0, width - 1)
    bbox[..., 1::2] = torch.clamp(bbox[..., 1::2], 0, height - 1)
    w, h = (bbox[..., 2:] - bbox[..., :2]).t()
    diag = torch.sqrt(w ** 2 + h ** 2)
    mask = (diag > min_box_diagonal) & (w > min_box_side) & (h > min_box_side)
    return mask


def filter_bboxes(detections: List[Dict[str, torch.Tensor]], height: int, width: int, min_box_diagonal: int = 30,
                  min_box_side: int = 20):
    filtered_bboxes = []
    for d in detections:
        bbox = d["boxes"]

        # first clamp boxes to image
        mask = diag_filter(bbox, height, width, min_box_diagonal, min_box_side)
        bbox = {k: v[mask] for k, v in d.items()}

        filtered_bboxes.append(bbox)

    return filtered_bboxes

def format_data(data, dataset="dsec", normalizer=None):
    if normalizer is None:
        normalizer = torch.stack([data.width[0], data.height[0], data.time_window[0]], dim=-1)
    if hasattr(data, "image"):
        data.image = data.image.float() / 255.0

    data.pos = torch.cat([data.pos, data.t.view((-1,1))], dim=-1)
    data.t = None
    data.x = data.x.float()
    data.pos = data.pos / normalizer

    # Print only the first 1-2 items
    """print(f"[DEBUG] Print in format data Dataset: {dataset}")
    print(f"  Normalizer (width, height, time_window): {normalizer}")
    print(f"  pos min/max x: {data.pos[:,0].min().item()}/{data.pos[:,0].max().item()}")
    print(f"  pos min/max y: {data.pos[:,1].min().item()}/{data.pos[:,1].max().item()}")
    print(f"  pos min/max t: {data.pos[:,2].min().item()}/{data.pos[:,2].max().item()}")

    print("\n[DEBUG] --- In format_data ---")
    if hasattr(data, "bbox0"):
        print("bbox0 shape:", data.bbox0.shape)
        print("bbox0 (first 5):", data.bbox0[:5].tolist())
    else:
        print("No bbox0 in data")

    if hasattr(data, "pos"):
        print("pos shape:", data.pos.shape)
        print("pos (first 5):", data.pos[:5].tolist())"""

    return data

def bbox_t_to_ndarray(bbox, t):
    dtype = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1')]
    if len(bbox) == 3:
        dtype.append(('class_confidence', '<f4'))

    boxes = bbox['boxes'].numpy()
    labels = bbox['labels'].numpy()

    output = np.zeros(shape=(len(boxes),), dtype=dtype)
    output['t'] = t
    output['x'] = boxes[:, 0]
    output['y'] = boxes[:, 1]
    output['w'] = boxes[:, 2] - boxes[:, 0]
    output['h'] = boxes[:, 3] - boxes[:, 1]
    output['class_id'] = labels

    if len(bbox) == 3:
        output['class_confidence'] = bbox["scores"].numpy()

    return output


def compile(detections, sequences, timestamps):
    output = {}
    for det, s, t in zip(detections, sequences, timestamps):
        if s not in output:
            output[s] = []
        output[s].append(bbox_t_to_ndarray(det, t))

    if len(output) > 0:
        output = {k: np.concatenate(v) for k, v in output.items() if len(v) > 0}

    return output

def to_cpu(data_list: List[Dict[str, torch.Tensor]]):
    return [{k: v.cpu() for k, v in d.items()} for d in data_list]

class Buffer:
    def __init__(self):
        self.buffer = []

    def extend(self, elements: List[Dict[str, torch.Tensor]]):
        self.buffer.extend(to_cpu(elements))

    def clear(self):
        self.buffer.clear()

    def __iter__(self):
        return iter(self.buffer)

    def __next__(self):
        return next(self.buffer)



class DetectionBuffer:
    def __init__(self, height: int, width: int, classes: List[str]):
        self.height = height
        self.width = width
        self.classes = classes
        self.detections = Buffer()
        self.ground_truth = Buffer()

    def compile(self, sequences, timestamps):
        detections = compile(self.detections, sequences, timestamps)
        groundtruth = compile(self.ground_truth, sequences, timestamps)
        return detections, groundtruth

    def update(self, detections: List[Dict[str, torch.Tensor]], groundtruth: List[Dict[str, torch.Tensor]], dataset: str, height=None, width=None):
        self.detections.extend(detections)
        self.ground_truth.extend(groundtruth)

    def compute(self)->Dict[str, float]:
        output =  evaluate_detection(self.ground_truth.buffer, self.detections.buffer, height=self.height, width=self.width, classes=self.classes)
        output = {k.replace("AP", "mAP"): v for k, v in output.items()}
        self.detections.clear()
        self.ground_truth.clear()
        return output


class DictBuffer:
    def __init__(self):
        self.running_mean = None
        self.n = 0

    def __recursive_mean(self, mn: float, s: float):
        return self.n / (self.n + 1) * mn + s / (self.n + 1)

    def update(self, dictionary: Dict[str, float]):
        if self.running_mean is None:
            self.running_mean = {k: 0 for k in dictionary}

        self.running_mean = {k: self.__recursive_mean(self.running_mean[k], dictionary[k]) for k in dictionary}
        self.n += 1

    def save(self, path):
        torch.save(self.running_mean, path)

    def compute(self)->Dict[str, float]:
        return self.running_mean

class PixelErrorBuffer:
    def __init__(self, device="cpu"):
        self.conf_threshold=0.1                                        #set to 0.001 instead bc conf will go to 0.0 for better evaluation of fp
        self.device = device
        self.all_errors = []
        self.fn=0
        self.fp=0
        self.tp=1
        self.use_distance = False

    def update(self, detections, targets):
        for det, gt in zip(detections, targets):
            error = self.compute_pixel_error(det, gt)
            if error is not None:
                self.all_errors.append(error)

    def compute_pixel_error(self, detections, targets):
        det_boxes = detections["boxes"].to(self.device)
        det_scores = detections["scores"].to(self.device)

        #filter out low conf detections
        mask = det_scores > self.conf_threshold
        det_boxes = det_boxes[mask]
        det_scores = det_scores[mask]

        gt_boxes = targets["boxes"].to(self.device)
        if gt_boxes.size(0) > 1:
            raise ValueError("more than one ground truth for ball data not valid")

        #compute FP or FN
        if det_boxes.numel() == 0 and gt_boxes.numel() > 0:
            self.fn+= 1
            return None
        if det_boxes.numel() > 0 and gt_boxes.numel() == 0:
            self.fp+= det_boxes.size(0)
            return None

        if self.use_distance:
            cx_det = (det_boxes[:, 0] + det_boxes[:, 2]) / 2
            cy_det = (det_boxes[:, 1] + det_boxes[:, 3]) / 2
            cx_gt = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
            cy_gt = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

            distances = torch.sqrt((cx_det - cx_gt)**2 + (cy_det - cy_gt)**2)
            valid_mask = distances < 20
            if valid_mask.any():
                det_boxes = det_boxes[valid_mask]
                det_scores = det_scores[valid_mask]
                distances = distances[valid_mask]
            else:
                # all detections too far 
                self.fn += 1
                self.fp += det_boxes.size(0)
                return None

            best_idx = torch.argmax(det_scores)
            best_box = det_boxes[best_idx].unsqueeze(0)
            errors = distances[best_idx].unsqueeze(0) 
            self.fp = det_boxes.size(0) - 1
            return errors.mean()

        sorted_idx = torch.argsort(det_scores, descending=True) 
        det_boxes = det_boxes[sorted_idx]

        #choose box with best score
        best_box=det_boxes[0].unsqueeze(0)
  
        self.fp = det_boxes.size(0) - 1                     #unmatched pred are FP

        #compute pixel error for matched pred and gt after selecting by score
        cx_pred = (best_box[:, 0] + best_box[:, 2]) / 2
        cy_pred = (best_box[:, 1] + best_box[:, 3]) / 2
        cx_gt = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        cy_gt = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

        error = torch.sqrt((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2)

        return error

    def compute(self):
        if not self.all_errors:
            return {"mean_pixel_error": None, "std_pixel_error": None}

        all_errors = torch.cat(self.all_errors)
        precision=self.tp/(self.fp+self.tp)
        return {
            "mean_pixel_error": all_errors.mean().item(),
            "std_pixel_error": all_errors.std(unbiased=False).item(),
            "precision": precision,
            "FP": self.fp,
            "FN": self.fn
        }


"""        
#save for dsec
class PixelErrorBuffer:
    def __init__(self, iou_threshold=0.5, device="cpu"):
        self.iou_threshold = iou_threshold
        self.conf_threshold=0.3
        self.device = device
        self.all_errors = []
        self.fn=0
        self.fp=0
        self.tp=0

    def update(self, detections, targets):
        for det, gt in zip(detections, targets):
            mean_pe, std_pe, errors = self.compute_pixel_error(det, gt)
            if errors is not None:
                self.all_errors.append(errors)

    def compute_pixel_error(self, detections, targets):
        det_boxes = detections["boxes"].to(self.device)
        det_scores = detections["scores"].to(self.device)

        #filter out low conf detections
        mask = det_scores > conf_threshold
        det_boxes = det_boxes[mask]
        det_scores = det_scores[mask]

        gt_boxes = targets["boxes"].to(self.device)

        if det_boxes.numel() == 0 and gt_boxes.numel() == 0:
            print("No detections and no ground truth exist.")
            return None, None, None

        #compute FP or FN
        if det_boxes.numel() == 0 and gt_boxes.numel() > 0:
            self.fn+= gt_boxes.size(0)
            return None, None, None
        if det_boxes.numel() > 0 and gt_boxes.numel() == 0:
            self.fp+= det_boxes.size(0)
            return None, None, None

       # sort predictions by score 
        sorted_idx = torch.argsort(det_scores, descending=True) 
        det_boxes = det_boxes[sorted_idx]

        #IoU based matching to choose best pred-gt pair
        ious = bboxes_iou(det_boxes, gt_boxes, xyxy=True) # [num_preds, num_gts]
        matched_det, matched_gt = [], []

        #iterating through det_boxes
        for d in range(det_boxes.size(0)):
            best_iou, best_gt = torch.max(ious[d], dim=0)                               #check for detection d which gt has best iou
            if best_iou >= self.iou_threshold and best_gt.item() not in matched_gt:
                matched_det.append(d)
                matched_gt.append(best_gt.item())

        if len(matched_det) == 0:
            self.fp += det_boxes.size(0)
            self.fn += gt_boxes.size(0)
            return None, None, None
        else:
            self.tp=len(matched_det)

        if len(matched_gt) < gt_boxes.numel():                                  #unmatched gt are FN
            self.fn= gt_boxes.numel() - len(matched_gt)                       
        if len(matched_det) < det_boxes.numel():                              #unmatched pred are FP
            self.fp =  det_boxes.numel() -len(matched_det)

        matched_det_boxes = det_boxes[matched_det]
        matched_gt_boxes = gt_boxes[matched_gt]

        #compute pixel error for matched pred and gt after selecting by score and iou
        cx_pred = (matched_det_boxes[:, 0] + matched_det_boxes[:, 2]) / 2
        cy_pred = (matched_det_boxes[:, 1] + matched_det_boxes[:, 3]) / 2
        cx_gt = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2
        cy_gt = (matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2

        errors = torch.sqrt((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2)

        return errors.mean(), errors.std(unbiased=False), errors

    def compute(self):
        if not self.all_errors:
            return {"mean_pixel_error": None, "std_pixel_error": None}

        all_errors = torch.cat(self.all_errors)
        precision=self.tp/(self.fp+self.tp)
        return {
            "mean_pixel_error": all_errors.mean().item(),
            "std_pixel_error": all_errors.std(unbiased=False).item(),
            "precision": precision,
            "FP": self.fp,
            "FN": self.fn
        }"""