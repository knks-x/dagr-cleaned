import argparse
import yaml

from pathlib import Path


def BASE_FLAGS():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--dataset_directory', type=Path, default=argparse.SUPPRESS, help="Path to the directory containing the dataset.")
    parser.add_argument('--output_directory', type=Path, default=argparse.SUPPRESS, help="Path to the logging directory.")
    parser.add_argument("--checkpoint", type=Path, default=argparse.SUPPRESS, help="Path to the directory containing the checkpoint.")
    parser.add_argument("--img_net", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--img_net_checkpoint", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--sync", type=str, default=argparse.SUPPRESS, help="Choose sync mode front or back")


    parser.add_argument("--config", type=Path, default="../config/detection.yaml")
    parser.add_argument("--use_image", action="store_true")
    parser.add_argument("--no_events", action="store_true")
    parser.add_argument("--pretrain_cnn", action="store_true")
    parser.add_argument("--keep_temporal_ordering", action="store_true")
    #parser.add_argutment("--use_split", action="store_true")

    # task params
    parser.add_argument("--task", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--dataset", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--crop_to_dsec", action="store_true")

    # graph params
    parser.add_argument('--radius', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--time_window_us', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--max_neighbors', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--n_nodes', default=argparse.SUPPRESS, type=int)

    # learning params
    parser.add_argument('--batch_size', default=argparse.SUPPRESS, type=int)

    # network params
    parser.add_argument("--activation", default=argparse.SUPPRESS, type=str, help="Can be one of ['Hardshrink', 'Hardsigmoid', 'Hardswish', 'ReLU', 'ReLU6', 'SoftShrink', 'HardTanh']")
    parser.add_argument("--edge_attr_dim", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--aggr", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--kernel_size", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--pooling_aggr", default=argparse.SUPPRESS, type=str)

    parser.add_argument("--base_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument("--after_pool_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument('--net_stem_width', default=argparse.SUPPRESS, type=float)
    parser.add_argument("--yolo_stem_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument("--num_scales", default=argparse.SUPPRESS, type=int)
    parser.add_argument('--pooling_dim_at_output', default=argparse.SUPPRESS)
    parser.add_argument('--weight_decay', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--clip', default=argparse.SUPPRESS, type=float)

    parser.add_argument('--aug_p_flip', default=argparse.SUPPRESS, type=float)

    return parser

def FLAGS():
    parser = BASE_FLAGS()

    # learning params
    parser.add_argument('--aug_trans', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--aug_zoom', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--exp_name', default=argparse.SUPPRESS, type=str)
    parser.add_argument('--l_r', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--no_eval', action="store_true")
    parser.add_argument('--tot_num_epochs', default=argparse.SUPPRESS, type=int)

    parser.add_argument('--run_test', action="store_true")

    parser.add_argument('--num_interframe_steps', type=int, default=10)

    # visualize detections params
    parser.add_argument("--detections_folder", help="Path to folder with detections.", type=Path)
    parser.add_argument("--dataset_directory_vis", help="Path to DSEC folder including which split.", type=Path, default="/data/scratch1/daniel/datasets/DSEC_fragment/test")
    parser.add_argument("--vis_time_step_us", help="Number of microseconds to step each iteration.", type=int, default=1000)
    parser.add_argument("--event_time_window_us", help="Length of sliding event time window for visualization.", type=int, default=5000)
    parser.add_argument("--sequence", help="Sequence to visualize. Must be an official DSEC sequence e.g. zurich_city_13_b", default="zurich_city_13_b", type=str)
    parser.add_argument("--write_to_output", help="Whether to save images in folder ${detections_folder}/visualization. Otherwise, just cv2.imshow is used.", action="store_true")

    #remove classification task
    parser.add_argument("--no_cls", action="store_true")

    args = parser.parse_args()

    if args.config != "":
        args = parse_config(args, args.config)

    args.dataset_directory = Path(args.dataset_directory)
    args.dataset_directory_vis = Path(args.dataset_directory_vis)
    args.output_directory = Path(args.output_directory)

    #if "checkpoint" in args:
    #    args.checkpoint = Path(args.checkpoint)

    return args

def FLOPS_FLAGS():
    parser = BASE_FLAGS()

    # for flop eval
    parser.add_argument("--check_consistency", action="store_true")
    parser.add_argument("--dense", action="store_true")

    # for runtime eval
    args = parser.parse_args()

    if args.config != "":
        args = parse_config(args, args.config)

    args.dataset_directory = Path(args.dataset_directory)
    args.output_directory = Path(args.output_directory)

    if "checkpoint" in args:
        args.checkpoint = Path(args.checkpoint)

    return args


def parse_config(args: argparse.ArgumentParser, config: Path):
    with config.open() as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        for k, v in config.items():
            if k not in args:
                setattr(args, k, v)
        return args
