from pathlib import Path 
import cv2
import numpy as np
from dsec_det.directory import DSECDirectory

from dsec_det.preprocessing import compute_img_idx_to_track_idx
from dsec_det.io import extract_from_h5_by_timewindow
from dsec_det.visualize import render_object_detections_on_image, render_events_on_image
from dsec_det.label import CLASSES


class DSECDet:
    def __init__(self, root: Path, split: str="train", sync: str="back", debug: bool=False, split_config=None):
        """
        root: Root to the the DSEC dataset (the one that contains 'train' and 'test'
        split: Can be one of ['train', 'test']
        window_size: Number of microseconds of data
        sync: Can be either 'front' (last event ts), or 'back' (first event ts). Whether the front of the window or
              the back of the window is synced with the images.

        Each sample of this dataset loads one image, events, and labels at a timestamp. The behavior is different for 
        sync='front' and sync='back', and these are visualized below.

        Legend: 
        . = events
        | = image
        L = label

        sync='front'
        -------> time
        .......|
               L

        sync='back'
        -------> time
        |.......
               L
        
        """
        assert root.exists()
        assert split in ['train', 'test', 'val']
        #assert (root / split).exists()
        assert sync in ['front', 'back']

        self.debug = debug
        self.classes = CLASSES

        self.root = root / ("train" if split in ['train', 'val'] else "test")
        self.sync = sync

        self.height = 480
        self.width = 640

        self.directories = dict()
        self.img_idx_track_idxs = dict()

        if split_config is None:
            self.subsequence_directories = list(self.root.glob("*/"))
        else:
            available_dirs = list(self.root.glob("*/"))
            self.subsequence_directories = [self.root / s for s in split_config[split] if self.root / s in available_dirs]
        
        self.subsequence_directories = sorted(self.subsequence_directories, key=self.first_time_from_subsequence)

        for f in self.subsequence_directories:
            directory = DSECDirectory(f)
            self.directories[f.name] = directory
            self.img_idx_track_idxs[f.name] = compute_img_idx_to_track_idx(directory.tracks.tracks['t'],
                                                                           directory.images.timestamps)

    def first_time_from_subsequence(self, subsequence):
        return np.genfromtxt(subsequence / "images/timestamps.txt", dtype="int64")[0]

    def __len__(self):
        return sum(len(v)-1 for v in self.img_idx_track_idxs.values())

    def __getitem__(self, item):
        output = {}
        output['image'] = self.get_image(item)
        output['events'] = self.get_events(item)
        output['tracks'] = self.get_tracks(item)

        if self.debug:
            # visualize tracks and events
            events = output['events']
            image = (255 * (output['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8")
            output['debug'] = render_events_on_image(image, x=events['x'], y=events['y'], p=events['p'])
            output['debug'] = render_object_detections_on_image(output['debug'], output['tracks'])

        return output

    def get_index_window(self, index, num_idx, sync="back"):
        if sync == "front":
            assert 0 < index < num_idx                          
            i_0 = index - 1                                     
            i_1 = index
        else:
            #assert 0 <= index < num_idx - 1
            i_0 = index
            i_1 = np.clip(index + 1, 0, num_idx - 1)                #clip secures that index+1 is in range of indices, e.g. last index+1 out of bound -> uses last idx instead.

        return i_0, i_1

    def get_tracks(self, index, mask=None, directory_name=None):
        #start at i_0 -> get all tracks for the tupel at position i_0 (tracks within these frames)

        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)              #return local idx (tupel position), group detections per slice and sequence
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)          #checks with local idx and length of group detection stack what is the window to look at
                                                                                                        #i_0, i_1 are indices into the filtered image pairs, not frame numbers. i_0 selects tupel to fetch tracks for
        if self.sync=="front":
            idx0, idx1 = img_idx_to_track_idx[i_1]
        else:
            idx0, idx1 = img_idx_to_track_idx[i_0]                                                      #start and end indices in the track array correspond to the detections for the frame at position i_0.

        tracks = directory.tracks.tracks[idx0:idx1]                                                    #get all track detections that belong to that frame (or that frame’s interval)
        if mask is not None:
            tracks = tracks[mask[idx0:idx1]]

        return tracks

    def get_events(self, index, directory_name=None):
        #returns events inbetween the time stamps of tupel i_0 and i_1

        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)              #returns local index and local mapping of that index(e.g. image_index_0)
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        t_0, t_1 = directory.images.timestamps[[i_0, i_1]]
        events = extract_from_h5_by_timewindow(directory.events.event_file, t_0, t_1)
        return events

    def get_image(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        image_files = directory.images.image_files_distorted
        image = cv2.imread(str(image_files[index]))
        return image

    def rel_index(self, index, directory_name=None):
        if directory_name is not None:
            img_idx_to_track_idx = self.img_idx_track_idxs[directory_name]
            directory = self.directories[directory_name]
            return index, img_idx_to_track_idx, directory

        for f in self.subsequence_directories:
            img_idx_to_track_idx = self.img_idx_track_idxs[f.name]
            if len(img_idx_to_track_idx)-1 <= index:
                index -= (len(img_idx_to_track_idx)-1)
                continue
            else:
                return index, img_idx_to_track_idx, self.directories[f.name]
        else:
            raise ValueError
