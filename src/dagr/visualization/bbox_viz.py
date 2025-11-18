import numpy as np
import cv2
import torchvision
import torch


from dsec_det.label import COLORS, CLASSES

_COLORS = np.array([[0.000, 0.8, 0.1], [1, 0.67, 0.00]])
class_names = ["car", "pedestrian"]

GT_COLOR = (255, 0, 0)      # red 
PRED_COLOR = (0, 255, 0)    # green


def draw_bbox_on_img(img, x, y, w, h, labels, scores=None, conf=0.5, nms=0.45, label="", linewidth=2, filtered_boxes=False):
    if scores is not None:
        mask = filter_boxes(x, y, w, h, labels, scores, conf, nms)
        x = x[mask]
        y = y[mask]
        w = w[mask]
        h = h[mask]
        labels = labels[mask]
        scores = scores[mask]

    for i in range(len(x)):
        if scores is not None and scores[i] < conf:
            continue

        x0 = int(x[i])
        y0 = int(y[i])
        x1 = int(x[i] + w[i])
        y1 = int(y[i] + h[i])
        cls_id = int(labels[i])

        """uncomment when also logging labels/text in the images:

        if scores is not None:                                                  
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()   

        text = f"{label}-{class_names[cls_id]}"

        if scores is not None:
            text += f":{scores[i] * 100: .1f}"

        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        """

        cv2.rectangle(img, (x0, y0), (x1, y1), PRED_COLOR, linewidth)       #draw the box

        """
        txt_height = int(1.5*txt_size[1])
        cv2.rectangle(img, (x0, y0 - txt_height), (x0 + txt_size[0] + 1, y0 + 1), txt_bk_color, -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]-txt_height), font, 0.4, txt_color, thickness=1)
        """
    if filtered_boxes:
        dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('w', 'f4'), ('h', 'f4'), ('class_id', 'u1'), ('class_confidence', 'f4')])
        filtered_boxes = np.zeros(len(x), dtype=dtype)
        filtered_boxes['x'] = x
        filtered_boxes['y'] = y
        filtered_boxes['w'] = w
        filtered_boxes['h'] = h
        filtered_boxes['class_id'] = labels
        filtered_boxes['class_confidence'] = scores
        return img, filtered_boxes
    else:
        return img

def filter_boxes(x, y, w, h, labels, scores, conf, nms):
    mask = scores > conf

    x1, y1 = x + w, y + h
    box_coords = np.stack([x, y, x1, y1], axis=-1)

    nms_out_index = torchvision.ops.batched_nms(
        torch.from_numpy(box_coords),
        #torch.from_numpy(np.ascontiguousarray(scores)),                    #issue while using visualize detections + wandb
        torch.from_numpy(np.ascontiguousarray(scores).astype(np.float32)),
        torch.from_numpy(labels),
        nms
    )

    nms_mask = np.ones_like(mask) == 0
    nms_mask[nms_out_index] = True

    return mask & nms_mask


def draw_gtbox_on_img(img, x, y, w, h, labels, label="", linewidth=2):

    for i in range(len(x)):

        x0 = int(x[i])
        y0 = int(y[i])
        x1 = int(x[i] + w[i])
        y1 = int(y[i] + h[i])
        cls_id = int(labels[i])

        #uncomment when also logging labels
        #text = f"{label}-{CLASSES[cls_id]}"
        if cls_id==0 or cls_id==2:
            COLOR=GT_COLOR
        else:
            COLOR=(200, 0, 0)

        #txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        #font = cv2.FONT_HERSHEY_SIMPLEX

        #txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), GT_COLOR, linewidth)
        #txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        #txt_height = int(1.5*txt_size[1])
        #cv2.rectangle(img, (x0, y0 - txt_height), (x0 + txt_size[0] + 1, y0 + 1), txt_bk_color, -1)
        #cv2.putText(img, text, (x0, y0 + txt_size[1]-txt_height), font, 0.4, txt_color, thickness=1)
    return img


"draw boxes without filtering + opacity of color wrt confidence level -- for debugging"

def draw_bbox_on_img_all_conf(img, x, y, w, h, labels, scores=None, label="", linewidth=2):
    for i in range(len(x)):
        cls_id = int(labels[i])
        base_color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()

        # Adjust color intensity based on confidence (0.0 -> very transparent, 1.0 -> full color)
        if scores is not None:
            alpha = float(scores[i])  # assuming scores in [0,1]
            color = [int(c * alpha) for c in base_color]
        else:
            color = base_color

        x0, y0 = int(x[i]), int(y[i])
        x1, y1 = int(x[i] + w[i]), int(y[i] + h[i])
        
        thickness = int(1 + 2*scores[i])
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)

        # Optional: draw confidence text
        if scores is not None:
            text = f"{label}-{class_names[cls_id]}:{scores[i]:.2f}"
            txt_color = (0, 0, 0) if np.mean(base_color) > 128 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (x0, y0 - 2), font, 0.4, txt_color, thickness=1)

    return img