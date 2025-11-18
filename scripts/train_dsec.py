# avoid matlab error on server
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import tqdm
import wandb
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams, log_bboxes, log_single_sample
from dagr.utils.buffers import DetectionBuffer, PixelErrorBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC

from train_balldata import debug_detections

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

def visualize_events(batch, sample_idx=0, save_path=None, title="Event Window"):
    mask = batch.batch == sample_idx  
    pos = batch.pos[mask].cpu().numpy()  # [N, 3] -> x,y,t
    p = batch.x[mask].cpu().numpy().squeeze()  # polarity
    
    x, y, t = pos.T
    colors = np.where(p > 0, "red", "blue")

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, s=1, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlim([0, 2])
    plt.ylim([0, 2])
    plt.title(title)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print("Saved!")
        plt.close()  # close to free memory
    else:
            plt.show()

def visualize_events_with_gt(batch, sample_idx=0, save_path=None, title="Loaded Events + GT"):
        """
        Visualize events + GT box for one sample inside a DataBatch.
        """
        mask = batch.batch == sample_idx
        pos = batch.pos[mask].cpu().numpy()
        p = batch.x[mask].cpu().numpy().squeeze()

        x, y, t = pos.T
        colors = np.where(p > 0, "red", "blue")

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c=colors, s=1, alpha=0.7)

        # bbox for this sample
        bbox = batch.bbox0[sample_idx].cpu().numpy()
        x0, y0, w, h, cat = bbox
        x0 /= batch.width
        y0 /= batch.height
        w  /= batch.width
        h  /= batch.height

        """ use this scale when balldata to dsec
        # bbox for this sample
        bbox = batch.bbox0[sample_idx].cpu().numpy()
        x0, y0, w, h, cat, conf = bbox
        x0 /= self.width_dsec // self.scale
        y0 /= self.height_dsec // self.scale
        w  /= self.width_dsec // self.scale
        h  /= self.height_dsec // self.scale"""

        rect = patches.Rectangle((x0, y0), w, h, linewidth=2,
                                edgecolor="lime", facecolor="none")
        plt.gca().add_patch(rect)

        plt.gca().invert_yaxis()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(title)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close()  # close to free memory
        else:
            plt.show()

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


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          epoch: int,
          run_name=""
          ):

    model.train()
    img_counter = 0
    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):               #tqdm: warps the trainloader and displays training progress in terminal. i=num of batch, data=num of sample
        data = data.cuda(non_blocking=True)                                                 #gpu
        data = format_data(data, dataset=args.dataset)

        if i % 10 == 0:                                                                    # only every 100th batch 
            print("visualizing events+boxes")
            #print_batch_info(data)
            save_path=f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/dsec/{img_counter}"                                           #take sample with idx 0 from the current batch
            visualize_events(data, sample_idx=0, save_path=save_path)
            #visualize_events_with_gt(data, sample_idx=0, save_path=save_path)
            img_counter += 1

        optimizer.zero_grad(set_to_none=True)
        

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}

        #debug gradients
        iou_loss = loss_dict.pop("iou_loss")
        iou_loss.backward(retain_graph=True)
        print("== Gradients from IoU loss ==")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name:30s} | grad_norm = {param.grad.data.norm(2):.6f}")
        optimizer.zero_grad(set_to_none=True)

        """conf_loss = loss_dict.pop("conf_loss")
        conf_loss.backward(retain_graph=True)
        print("\n== Gradients from Confidence loss ==")
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
         pixel_error=False,
         epoch=0):

    model.eval()

    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)
    pixel_error_calc = PixelErrorBuffer(device="cuda")

    class_names = loader.dataset.classes

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data, dataset=args.dataset)

        detections, targets = model(data)               #returned from DAGR.forward(data): appended detections+targets in same dict format

        #detections+targets are in x1,x2,y1,y2 format now
        debug_detections("Detections", detections)
        debug_detections("Targets", targets)

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
    dataset_path = args.dataset_directory / args.dataset

    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, sync=args.sync, debug=False,
                         min_bbox_diag=15, min_bbox_height=10)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, sync=args.sync, debug=False,
                        min_bbox_diag=15, min_bbox_height=10)


    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True) #follow_batch=['bbox', 'bbox0']:
    num_iters_per_epoch = len(train_loader)
    print("Num iterations per epoch:", num_iters_per_epoch)         #num of batches per epoch ~2256

    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    model = model.cuda()
    ema = ModelEMA(model)           #smoothing the model parameters, maybe like this: EMA weights = decay * EMA weights + (1 - decay) * current model weights

    nominal_batch_size = 64         
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

    lr_func = LRSchedule(warmup_epochs=.3,
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=output_directory,
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=args)

    start_epoch = 0
    if hasattr(args, "checkpoint"):        #before: "resume_checkpoint"
        start_epoch = checkpointer.restore_checkpoint(args.checkpoint, best=False)  #before: args.resume_checkpoint
        print(f"Resume from checkpoint at epoch {start_epoch}")
    else:
        print("Start training from scratch")

    with torch.no_grad():               #pre-testing, only for two steps
        mapcalc, pixel_error_calc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset, pixel_error=True)
        mapcalc.compute()
        pixel_error_calc.compute()

    print("starting to train")
    for epoch in range(start_epoch, args.tot_num_epochs):
        print(epoch)

        # Training
        train(train_loader, model, ema, lr_scheduler, optimizer, args, epoch, run_name=wandb.run.name)
        checkpointer.checkpoint(epoch, name=f"last_model")

        # Evaluate every 3 epochs
        if epoch % 3 == 0:
            with torch.no_grad():
                print("evaluating")
                mapcalc, pixel_error_calc = run_test(test_loader, ema.ema, dataset=args.dataset, epoch=epoch, pixel_error=True)
                
                #mAP
                metrics = mapcalc.compute()
                checkpointer.process(metrics, epoch)

                #pixel error
                pixel_error_metrics = pixel_error_calc.compute()
                print("Final Pixel Error:", pixel_error_metrics)
                #log to wandb
                if pixel_error_metrics["mean_pixel_error"] is not None:
                    mean_pixel_error = pixel_error_metrics["mean_pixel_error"]
                    std_pixel_error = pixel_error_metrics["std_pixel_error"]
                    upper = mean_pixel_error + std_pixel_error
                    lower = mean_pixel_error - std_pixel_error
                    precision=pixel_error_metrics["precision"]
                    fp=pixel_error_metrics["FP"]

                    wandb.log({
                        "pixel_err/mean": mean_pixel_error,
                        "pixel_err/std": std_pixel_error,
                        #"pixel_err/upper": upper,
                        #"pixel_err/lower": lower,
                        "pixel_err/precision": precision,
                        "pixel_err/FP": fp,
                        "epoch": epoch
                        })
        else: print(f"No predictions or gt boxes found. Can't compute pixel error.")
                
                torch.cuda.empty_cache()


