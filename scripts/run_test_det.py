# avoid matlab error on server
import os
import torch
import wandb
import numpy as np
import random
import datetime

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from torch_geometric.data import DataLoader
from dagr.utils.args import FLAGS

from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

from dagr.utils.logging import set_up_logging_directory, log_hparams
from dagr.utils.testing import run_test_with_visualization

from dagr.utils.seeding import seed_everything


if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np
    
    #true for deterministic seeding 
    seed_det=True

    if seed_det:
       seed = 42
       seed_everything(seed)

    else:
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
    dataset_path = args.dataset_directory.parent / args.dataset

    test_dataset = DSEC(args.dataset_directory, "test", Augmentations.transform_testing, sync=args.sync, debug=False, min_bbox_diag=15, min_bbox_height=10)
    print("Len Test dataset:", len(test_dataset))
    
    #TODO: save mapping if necessary, with npy dicts instead of json. check which number of the tupel is the current/predicted frame (i_0,i_1).
    save_mapping=False
    if save_mapping:
        import json

        subset_mapping = []

        for i in range(len(test_dataset)):                          # i is iterating over image(including all)
            # resolve sequence, local index, images
            directory, image_index_pairs, track_mask, local_idx, seq_name = test_dataset.rel_index(i, r_sequence=True)
            image_index_0, image_index_1 = image_index_pairs[local_idx]

            # timestamps
            timestamp_0, timestamp_1 = test_dataset.dataset.images.timestamps[[image_index_0, image_index_1]]
            image_path_0 = str(test_dataset.dataset.images[image_index_0])

            # GT boxes
            detections_0 = test_dataset.dataset.get_tracks(image_index_0, mask=track_mask, directory_name=directory.root.name)

            # prev / next indices in the original dataset
            prev_idx = image_index_pairs[local_idx - 1][0] if local_idx > 0 else None
            next_idx = image_index_pairs[local_idx + 1][0] if local_idx < len(image_index_pairs) - 1 else None

            entry = {
                "subset_idx": i,
                "sequence": seq_name,
                "local_idx": local_idx,
                "image_index_0": int(image_index_0),
                "image_index_1": int(image_index_1),
                "timestamp_0": int(timestamp_0),
                "timestamp_1": int(timestamp_1),
                "image_path": image_path_0,
                "gt_boxes": tracks_to_array(detections_0),
                "prev_original_idx": int(prev_idx) if prev_idx is not None else None,
                "next_original_idx": int(next_idx) if next_idx is not None else None
            }
            subset_mapping.append(entry)

        # save JSON
        with open("subset_mapping.json", "w") as f:
            json.dump(subset_mapping, f, indent=2)


    num_iters_per_epoch = 1

    #random sampler - normally use this
    sampler = np.random.permutation(np.arange(len(test_dataset)))
    #no randomness
    sampler_det=list(range(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler_det, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()
    ema = ModelEMA(model)                       #ema wrapper for stable weights in eval/inference

    torch.cuda.empty_cache()

    assert "checkpoint" in args
    checkpoint = torch.load(args.checkpoint)
    ema.ema.load_state_dict(checkpoint['ema'])
    ema.ema.cache_luts(radius=args.radius, height=test_dataset.height, width=test_dataset.width)

    with torch.no_grad():
        metrics = run_test_with_visualization(test_loader, ema.ema, dataset=args.dataset)
        log_data = {f"testing/metric/{k}": v for k, v in metrics.items()}
        wandb.log(log_data)
        print(metrics['mAP'])



