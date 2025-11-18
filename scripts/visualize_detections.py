import cv2
import argparse
import wandb

from pathlib import Path
import numpy as np

from dsec_det.directory import DSECDirectory
from dsec_det.io import extract_from_h5_by_timewindow, extract_image_by_index, load_start_and_end_time
from dsec_det.preprocessing import compute_index

from dagr.visualization.bbox_viz import draw_bbox_on_img
from dagr.visualization.event_viz import draw_events_on_image

from dagr.utils.logging import set_up_logging_directory
from dagr.utils.args import FLAGS


if __name__ == '__main__':

    args = FLAGS()
    global_step = 0 

    assert args.dataset_directory.exists()
    assert args.vis_time_step_us > 0
    assert args.event_time_window_us > 0

    if args.write_to_output:
        assert (args.detections_folder / f"detections_{args.sequence}.npy").exists()
        assert args.detections_folder.exists()
        output_path = args.detections_folder / "visualization"
        output_path.mkdir(parents=True, exist_ok=True)

    dsec_directory = DSECDirectory(args.dataset_directory / args.sequence)
    
    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)

    t0, t1 = load_start_and_end_time(dsec_directory)

    vis_timestamps = np.arange(t0, t1, step=args.vis_time_step_us)
    step_index_to_image_index = compute_index(dsec_directory.images.timestamps, vis_timestamps)

    show_detections = args.detections_folder is not None

    if not show_detections:
        print("Did not specifiy detections. Just showing events and images.")

    if show_detections:
        detections_file = args.detections_folder / f"detections_{args.sequence}.npy"
        detections = np.load(detections_file)
        detection_timestamps = np.unique(detections['t'])
        step_index_to_boxes_index = compute_index(detection_timestamps, vis_timestamps)

    scale = 2

    for step, t in enumerate(vis_timestamps):

        # find most recent image
        image_index = step_index_to_image_index[step]
        image = extract_image_by_index(dsec_directory.images.image_files_distorted, image_index)

        # find events within time window [image_timestamps, t]
        events = extract_from_h5_by_timewindow(dsec_directory.events.event_file, t-args.event_time_window_us, t)
        image = draw_events_on_image(image, events['x'], events['y'], events['p'])

        if show_detections:
            # find most recent bounding boxes
            boxes_index = step_index_to_boxes_index[step]
            boxes_timestamp = detection_timestamps[boxes_index]
            boxes = detections[detections['t'] == boxes_timestamp]

            # draw them on one image
            scale = 2
            image = draw_bbox_on_img(image, scale*boxes['x'], scale*boxes['y'], scale*boxes['w'], scale*boxes["h"],
                                     boxes["class_id"], boxes['class_confidence'], conf=0.3, nms=0.65)

        #log to wandb
        if show_detections:
            low_conf = np.any(boxes['class_confidence'] < 0.3)
            if len(boxes) == 0 or len(boxes) > 3 or low_conf:
                conf_str = ", ".join([f"{c:.2f}" for c in boxes['class_confidence']])
                wandb.log({f"{args.sequence}_odd_detection_frame": wandb.Image(image, caption=f"Step {step} | Conf: [{conf_str}]")}, step=global_step)
                global_step += 1


        if args.write_to_output:
            cv2.imwrite(str(output_path / ("%06d.png" % step)), image) 
        #else:
            #cv2.imshow("DSEC Det: Visualization", image)               #issue bc remote server has no GUI window - changed to:
            #cv2.imwrite("DSEC_Detection_Visualization.png", image)     #saves locally but overwrites
            #cv2.waitKey(3)




        

