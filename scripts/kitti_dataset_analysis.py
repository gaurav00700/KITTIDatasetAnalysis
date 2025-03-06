import os, sys
parent_dir = os.path.abspath(os.path.join(__file__ ,"../.."))
sys.path.insert(0, parent_dir) #add package path to python path
import numpy as np
import argparse
import open3d as o3d
import json
import tqdm
from collections import defaultdict
from typing import Literal, Union, List
from lib import tools, visualization
from lib import utils_kitti as utils
from lib.kitti_loader import KITTILoader

def add_parsers(args_list:list= None):
    """Method for creating argument parser"""

    parser = argparse.ArgumentParser(description='Parameters for dataset analysis')

    parser.add_argument('--dataset_path', type=str, 
                        default= "/mnt/c/Users/Gaurav/Downloads/Datasets/kitti/2011_09_26",
                        help='Path to dataset containing Sequences dataset')
    parser.add_argument('--seq_name', type=str, 
                        default= "0009",
                        help='Name of sequence')
    parser.add_argument('--filter_frame', type=int, 
                        nargs= '+', default=list(range(10,20)),
                        help='List of frames to filter')
    parser.add_argument('--data_io_path', type=str, 
                        default='data/output/Kitti',
                        help='path to save output data')
    return parser.parse_args(args_list) if args_list else parser.parse_args() 

def main(save_data:bool= False):
    #get argument parsers
    args = add_parsers()
    frame_ids=args.filter_frame
    kitti = KITTILoader(kitti_root=args.dataset_path, seq_name="0009")

    # Prepare dirs
    if not os.path.exists(args.data_io_path): 
        os.makedirs(args.data_io_path)

    # Get the file paths of lidar, camera and gps data
    cam_02_imgs = kitti.load_images(camera_name='02',frame_ids=frame_ids)
    pcds = kitti.load_pcds(frame_ids=frame_ids)
    
    # Get trackets (annotations) details in Lidar frame
    tracklet_rects, tracklet_types = kitti.load_tracklets(frame_ids=frame_ids)

    # Load camera
    K , T_velo_cam2, T_velo_img2 = kitti.load_camera(camera_name='02')

    # Script for saving the annotations in json ...
    annotation_dict = defaultdict(dict)
    imgdata_dict = defaultdict(dict)
    for i, frame_key in tqdm.tqdm(enumerate(tracklet_rects), total= len(tracklet_rects), desc= 'Loading img and bbox'):

        # Get annotations in Lidar frame
        bboxes_velo = tracklet_rects[frame_key]
        labels_frame = tracklet_types[frame_key]

        # Transform bboxes_frame from Lidar frame to image frame
        bboxes_img = []
        if len(bboxes_velo) > 0:
            for bbox_velo, label in zip(bboxes_velo, labels_frame):
                bbox_velo = np.row_stack((bbox_velo, np.ones(bbox_velo.shape[1])))
                bbox_img = np.dot(T_velo_img2[:3,:], bbox_velo) # 3x8
                bbox = (bbox_img/bbox_img[2:3,:])[:2] # Normalize by z, 3x8 -> 2x8
                x0, y0 = bbox[0].min(), bbox[1].min()
                x1, y1 =  bbox[0].max(), bbox[1].max()
                cx, xy, w, h = x0 + (x0+x1)/2, y0 + (y0+y1)/2, x1-x0, y1-y0
                bbox_ = {
                    'pos_xywh': [cx, xy, w, h],
                    'pos_xyxy': [x0, y0, x1, y1],
                    'pos_xy_2x8': bbox,
                    'cls_name': label,
                        }
                bboxes_img.append(bbox_) # 2x8

        annotation_dict[frame_key]['bboxes'] = bboxes_img   # bbox image frame
        imgdata_dict[frame_key]['img_data'] = cam_02_imgs[i]    # image

    # Visualize the image and annotations
    visualization.viz_img_bbox(annotation_dict, imgdata_dict, args.data_io_path)

    # Save data to json file
    if save_data:
        save_path = os.path.join(args.data_io_path, f"annotations_2d_Cam02.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(annotation_dict, f, indent=4 , cls=tools.NpEncoder)
            print(f"Annotation data is saved at:{save_path}")
    
if __name__ == '__main__':
    main(save_data=True) 





