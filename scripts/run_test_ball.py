# avoid matlab error on server
import os
import torch
import wandb
from pathlib import Path

import datetime
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch_geometric.data import DataLoader
from dagr.utils.args import FLAGS

from dagr.data.ball_data import BallDataset
from dagr.data.augment import Augmentations

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

from dagr.utils.logging import set_up_logging_directory, log_hparams
from dagr.utils.testing import run_test_with_visualization, run_test_without_visualization, run_test_pe


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

    #added this to have unique exp_name: use flag if given else timestamp
    exp_name = args.exp_name or  datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name)         #sets up dir: creates folder in FO and wandb run

    project = f"low_latency-{args.dataset}-{args.task}"
    print(f"PROJECT: {project}")                                                                        #prints all the model features
    log_hparams(args)

    print("init datasets")
    dataset_path = Path(args.dataset_directory)  / args.dataset / "data_cropped"
    split_dir = args.split_directory
    
    #set ball radius depending on dataset
    if args.dataset=="ball-inc":
        ball_diameter_min=35
        ball_diameter_max=45
    else:
        ball_diameter_min=40
        ball_diameter_max=40

    test_dataset = BallDataset(root=dataset_path, split="test", transform=None, debug=False,
                        ball_diameter_min=ball_diameter_min, ball_diameter_max=ball_diameter_max, split_dir=split_dir, crop_to_dsec=args.crop_to_dsec)
    num_iters_per_epoch = 1

    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()
    ema = ModelEMA(model)                       #ema wrapper for stable weights in eval/inference

    assert "checkpoint" in args
    checkpoint = torch.load(args.checkpoint)
    ema.ema.load_state_dict(checkpoint['ema'])
    ema.ema.cache_luts(radius=args.radius, height=test_dataset.height, width=test_dataset.width)

    only_pe=False

    with torch.no_grad():
        if only_pe:
            pixel_error_metrics=run_test_pe(test_loader, ema.ema, dataset=args.dataset)
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
            else: 
                print(f"No predictions or gt boxes found. Can't compute pixel error.")

        else:
            #run_test_with_visualization possible 
            metrics, pixel_error_metrics = run_test_without_visualization(test_loader, ema.ema, dataset=args.dataset)
            
            #mAP
            log_data = {f"testing/metric/{k}": v for k, v in metrics.items()}
            wandb.log(log_data)
            print("Final mAP:", metrics['mAP'])

            #pixel error
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
                            "testing/pixel_err/mean": mean_pixel_error,
                            "testing/pixel_err/std": std_pixel_error,
                            "testing/pixel_err/precision": precision,
                            "testing/pixel_err/FP": fp
                            })
            else: 
                print(f"No predictions or gt boxes found. Can't compute pixel error.")
                print("Final Mean Pixel Error:", pixel_error_metrics["mean_pixel_error"])
