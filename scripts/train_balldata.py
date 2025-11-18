# avoid matlab error on server
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import tqdm
import wandb
import math
import csv
import json

from pathlib import Path
import argparse

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams, log_bboxes, log_single_sample
from dagr.utils.buffers import DetectionBuffer, PixelErrorBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.ball_data import BallDataset

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

from dagr.data.ball.create_splits import average_map, plot_heatmap, compare_trajectories

def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad, nan=0.0)

def print_batch_info(batch):
    """
    Print which sequences and time windows are inside a DataBatch.
    """
    seqs = batch.sequence
    t0s = batch.t0
    t1s = batch.t1

    print("\n[Batch Info]")
    for i, (seq, t0, t1) in enumerate(zip(seqs, t0s, t1s)):
        print(f"  Window {i:02d}: sequence={seq},  t0={t0},  t1={t1},  Δt={t1 - t0}")


def debug_detections(name, detections, max_images=2, max_boxes=3):
    print(f"{name}: list with {len(detections)} images")
    for i, det in enumerate(detections[:max_images]):  # only first few images
        print(f"  Image {i}:")
        for k, v in det.items():
            if torch.is_tensor(v):
                print(f"    {k}: shape={tuple(v.shape)}")
                if v.numel() > 0:
                    print(f"      first {max_boxes} entries: {v[:max_boxes].tolist()}")
            else:
                print(f"    {k}: {type(v)}")


def save_detections_csv(target, predictions, filepath):
    #convert predictions from x1,y1,x2,y2 -> x1,y1,w,h
    preds_w = predictions[:, 2] - predictions[:, 0]
    preds_h = predictions[:, 3] - predictions[:, 1]
    preds_cx = predictions[:, 0] + preds_w/2
    preds_cy = predictions[:, 1] + preds_h/2
    preds = torch.stack([predictions[:, 0], predictions[:, 1], preds_w, preds_h, predictions[:, 4]], dim=1)
    t_w = target[2] - target[0]
    t_h = target[3] - target[1]
    t_cx = target[0] + t_w / 2
    t_cy = target[1] + t_h / 2
    t = torch.stack([t_cx, t_cy, t_w, t_h])

    with open(filepath, "w", newline="") as f:
        
        writer = csv.writer(f)
        writer.writerow(["cx", "cy", "w", "h", "conf"])
        writer.writerow(t.tolist())
        for row in preds:
            writer.writerow(row.tolist())


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          epoch: int,
          dataset: Dataset,
          run_name=""
          ):

    model.train()
    img_counter = 0
    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):               #tqdm: warps the trainloader and displays training progress in terminal. i=num of batch, data=num of sample
        data = data.cuda(non_blocking=True)                                                 #gpu
        data = format_data(data, args.dataset)

        if i % 10 == 0:                                                                    # only every 100th batch 
            print("visualizing events+boxes")
            save_path=f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/loaded_events+gt/{img_counter}"                                           #take sample with idx 0 from the current batch
            #dataset.visualize_events_with_gt_and_centers(data, sample_idx=0, save_path=save_path)
            dataset.visualize_events(data, sample_idx=0, save_path=save_path)
            #img_counter += 1
        
        optimizer.zero_grad(set_to_none=True)
        
        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}

        #debug gradients
        """if i % 500 == 0: 
            iou_loss = loss_dict.pop("iou_loss")
            iou_loss.backward(retain_graph=True)
            print("== Gradients from IoU loss ==")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name:30s} | grad_norm = {param.grad.data.norm(2):.6f}")
            optimizer.zero_grad(set_to_none=True)

            conf_loss = loss_dict.pop("conf_loss")
            conf_loss.backward(retain_graph=True)
            print("== Gradients from Confidence loss ==")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name:30s} | grad_norm = {param.grad.data.norm(2):.6f}")
            optimizer.zero_grad(set_to_none=True)
            l2_loss = loss_dict.pop("l2_loss")
            l2_loss.backward(retain_graph=True)
            print("== Gradients from L2 loss ==")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name:30s} | grad_norm = {param.grad.data.norm(2):.6f}")
            optimizer.zero_grad(set_to_none=True)"""
        #debug end

        loss = loss_dict.pop("total_loss")
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

        fix_gradients(model)

        optimizer.step()
        scheduler.step()

        ema.update(model)

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        wandb.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs})

def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
         dataset="gen1",
         epoch: int=0,
         exp_name="temp"
         ):

    model.eval()

    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)
    pixel_error_calc = PixelErrorBuffer(device="cuda")

    class_names = loader.dataset.classes

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data, args.dataset)

        detections, targets = model(data)               #returned from DAGR.forward(data): appended detections+targets in same dict format; are in x1,x2,y1,y2 format now
        
        #debug_detections("Detections", detections)
        #debug_detections("Targets", targets)

        #DEBUG CONFIDENCE vs LOCATION
        if i == 30:                                                     #31 batches in val set
            det1 = detections[0]
            target1 = targets[0]
            pred_boxes = det1["boxes"]          # shape (175, 4)
            pred_scores = det1["scores"]        # shape (175,)

            preds = torch.cat([pred_boxes, pred_scores.unsqueeze(1)], dim=1)  # shape (175, 5)
            target_box = target1["boxes"].squeeze(0)  

            save_detections_csv(target_box, preds, f"outputs/validation/{exp_name}_csv_epoch{epoch}-batch{i}.png")

        if i % 10 == 0:
            torch.cuda.empty_cache()

        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])
        pixel_error_calc.update(detections, targets)

        if dry_run_steps > 0 and i == dry_run_steps:
            break

    return mapcalc, pixel_error_calc

if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    if not hasattr(args, "checkpoint"):
        print("No checkpoint in FLAGS")
    else:
        checkpoint_path = Path(args.checkpoint)
        print("Checkpoint after FLAGS:", args.checkpoint)
        print("Type:", type(args.checkpoint))
        print("Exists?", Path(args.checkpoint).exists())

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
    log_hparams(args)

    augmentations = Augmentations(args)

    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset / "data_cropped"
    split_dir = args.split_directory

    #set ball radius depending on dataset
    if args.dataset=="ball-inc":
        ball_diameter_min=35
        ball_diameter_max=45
    else:
        #TODO: check which radius is necessary
        ball_diameter_min=40
        ball_diameter_max=40

    train_dataset = BallDataset(root=dataset_path, split="train", transform=None, debug=False, 
                         ball_diameter_min=ball_diameter_min, ball_diameter_max=ball_diameter_max, split_dir=split_dir, crop_to_dsec=args.crop_to_dsec)
    val_dataset = BallDataset(root=dataset_path, split="val", transform=None, debug=False,
                        ball_diameter_min=ball_diameter_min, ball_diameter_max=ball_diameter_max, split_dir=split_dir, crop_to_dsec=args.crop_to_dsec)

    train_loader = DataLoader(train_dataset, follow_batch=['bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True) 
    num_iters_per_epoch = len(train_loader)
    print("Num iterations per epoch:", num_iters_per_epoch)         #num of batches per epoch ~2256 (dsec)

    sampler = np.random.permutation(np.arange(len(val_dataset)))
    test_loader = DataLoader(val_dataset, sampler=sampler, follow_batch=['bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=val_dataset.height, width=val_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    model = model.cuda()
    ema = ModelEMA(model)           #smoothing the model parameters, maybe like this: EMA weights = decay * EMA weights + (1 - decay) * current model weights

    nominal_batch_size = 64         
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

    #at least 7500 iterations in warm up
    warmup_iters = 850*16/args.batch_size                                                        #warm_up_iters depend on batch_size
    warmup_epochs = warmup_iters / num_iters_per_epoch

    print("Warmup epochs:", warmup_epochs)
    lr_func = LRSchedule(warmup_epochs=warmup_epochs,                                           #in dsec: 0.3; num of iterations 2550 (batchsize 16)
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=output_directory,
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, 
                                ema=ema,
                                args=args)

    start_epoch = 0
    if hasattr(args, "checkpoint"):        #before: "resume_checkpoint"
        start_epoch = checkpointer.restore_checkpoint(args.checkpoint, best=False)  #before: args.resume_checkpoint
        print(f"Resume from checkpoint at epoch {start_epoch}")
    else:
        print("Start training from scratch")

    with torch.no_grad():               #pre-testing, only for two steps
        mapcalc, pixel_error_calc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset, epoch=start_epoch, exp_name=args.exp_name)
        mapcalc.compute()
        pixel_error_calc.compute()
        print("pe:", pixel_error_calc.compute()) 
        print("acc:", mapcalc.compute)

    print("starting to train")
    for epoch in range(start_epoch, args.tot_num_epochs):
        print(epoch)

        # Training
        train(train_loader, model, ema, lr_scheduler, optimizer, args, epoch, run_name=wandb.run.name, dataset=train_dataset)
        checkpointer.checkpoint(epoch, name=f"last_model")

        # Evaluate every 3 epochs
        if epoch % 1 == 0:
            with torch.no_grad():
                print("evaluating")
                mapcalc, pixel_error_calc = run_test(test_loader, ema.ema, dataset=args.dataset, epoch=epoch, exp_name=args.exp_name)
                
                #mAP
                metrics = mapcalc.compute()
                checkpointer.process(metrics, epoch)

                #pixel error
                pixel_error_metrics = pixel_error_calc.compute()
                print("Final Pixel Error:", pixel_error_metrics)
                #log to wandb
                if pixel_error_metrics["mean_pixel_error"] is not None:
                            mean_pe = pixel_error_metrics["mean_pixel_error"]
                            std_pe = pixel_error_metrics["std_pixel_error"]
                            precision=pixel_error_metrics["precision"]
                            fp=pixel_error_metrics["FP"]

                            wandb.log({
                                "validation/pixel_err/mean": mean_pe,
                                "validation/pixel_err/std": std_pe,
                                "validation/pixel_err/precision": precision,
                                "validation/pixel_err/FP": fp,
                                "epoch": epoch
                                })
                else: 
                    print(f"No predictions or gt boxes found. Can't compute pixel error.")
                
                torch.cuda.empty_cache()


