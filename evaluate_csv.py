import csv
import torch

def load_detections_csv(filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if rows[0][0] == "cx":
        rows = rows[1:]

    target_row = [float(x) for x in rows[0]]
    target = torch.tensor(target_row)  

    preds_rows = rows[1:]
    preds = torch.tensor([[float(x) for x in row] for row in preds_rows])  
    return target, preds


#target, preds = load_detections_csv("/data/sbaumann/nature-gnn/dagr/outputs/validation/debug_confidence_epoch21-batch30.png")
#target, preds = load_detections_csv("/data/sbaumann/nature-gnn/dagr/outputs/validation/ball_10*iou_1*obj_l1_csv_epoch18-batch30.png")
target, preds = load_detections_csv("/data/sbaumann/nature-gnn/dagr/outputs/validation/ball_5*iou_1*obj_ignoreWConf-2_csv_epoch12-batch30.png")

pe = torch.sqrt((preds[:, 0] - target[0])**2 + (preds[:, 1] - target[1])**2) 
pe = torch.round(pe * 100) / 100
preds_with_pe = torch.cat([pe.unsqueeze(1), preds], dim=1)  # shape (N, 6)

#sort after pe
preds_sorted = preds_with_pe[pe.argsort()]  

#output_csv="/data/sbaumann/nature-gnn/dagr/outputs/sorted-after-pe/val-debug_confidence_epoch21-batch30.png"
#output_csv="/data/sbaumann/nature-gnn/dagr/outputs/new_scale2-sorted-after-pe/debug_confidence_epoch45-batch30.png"
output_csv="/data/sbaumann/nature-gnn/dagr/outputs/sorted-after-pe/ball_5*iou_1*obj_ignoreWConf-2_csv_epoch12-batch30.png"

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pe", "cx", "cy", "w", "h", "conf"])
    writer.writerow([f"{0.0:.3f}"] + [f"{x:.3f}" for x in target.tolist()])
    for row in preds_sorted.tolist():
        writer.writerow([f"{row[0]:.3f}"] + [f"{x:.3f}" for x in row[1:]])