import numpy as np
import numpy.linalg as la
import json
import tqdm
from collections import defaultdict
from lib.tools import *
import open3d as o3d
import src.parseTrackletXML as xmlParser

def timestamp_to_float(file_path):
    """
    convert timestamps from datetime to float using timestamp .txt file.
    """
    # Read .txt file
    datetime_data = read_lines(file_path)

    #convert string to time stamp in Y-M-D H:M:S.nsec eg: 2011-09-26 13:02:26.584402579
    # timestamp_datetime = [pd.to_datetime(dt, format='%Y-%m-%d %H:%M:%S.%f') for dt in datetime_data]
    timestamp_datetime = []
    for dt in datetime_data:
        try:
            timestamp_datetime.append(pd.to_datetime(dt, format='%Y-%m-%d %H:%M:%S.%f'))
        except ValueError:
            # Ignore lines that don't match the expected format
            continue

    # Convert timestamp from datetime to float eg: 1632854894.595247
    timestamp_float = np.asarray([datetime.timestamp(dt) for dt in timestamp_datetime])

    return timestamp_float

def Rt_velo_to_imu(config_file_path: str):
    """
    create Rt transformation matrix from velo to imu using "calib_imu_to_velo.txt"
    """
    ego_config = read_lines(config_file_path)

    R_data = np.asarray([np.float32(i) for i in ego_config[1].split()[1:]])
    t_data = np.asarray([np.float32(i) for i in ego_config[2].split()[1:]])

    # Calculate T matrix using R and t data
    T_mat_velo2imu = Rt_to_T(R_data, t_data)

    return T_mat_velo2imu

def get_trans_imu_seq(gps_files:list)->list:
    """
    get transformations from vehicle to world frame for all frames (sequence)
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None
    T_w_imu_frames = list()
    for file in tqdm.tqdm (gps_files, desc="Loading data ..."):
        gps_data = np.loadtxt(file)
        lat, lon, alt, roll, pitch, yaw = gps_data[0], gps_data[1], gps_data[2], gps_data[3], gps_data[4], gps_data[5]
        
        # get R and t using gps dta
        if scale is None:
            scale = np.cos(lat*np.pi/180.)
        
        R, t = Rt_from_gps(lat, lon, roll, pitch, scale, alt, yaw) # All angles should be in radians

        if origin is None:
            origin = t

        # get T matrix using R and t
        T_w_imu = Rt_to_T(R, t - origin)
        # T_w_imu = Rt_to_T(R, origin-t)
        T_w_imu_frames.append(T_w_imu)
    
    return T_w_imu_frames

def skew_sym_matrix(u):
    return np.array([[    0, -u[2],  u[1]], 
                     [ u[2],     0, -u[0]], 
                     [-u[1],  u[0],    0]])

def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
        np.sin(angle) * skew_sym_matrix(axis) + \
        (1 - np.cos(angle)) * np.outer(axis, axis)

def read_bounding_boxes(file_name_bboxes):
    """
    Read the bounding boxes corresponding to the frame.
    """
    # open the file
    # with open (file_name_bboxes, 'r') as f:
    #     bboxes = json.load(f)
    bboxes = load_json(file_name_bboxes)
        
    boxes = [] # a list for containing bounding boxes  
    
    for bbox in bboxes.keys():
        bbox_read = {} # a dictionary for a given bounding box
        bbox_read['class'] = bboxes[bbox]['class']
        bbox_read['truncation']= bboxes[bbox]['truncation']
        bbox_read['occlusion']= bboxes[bbox]['occlusion']
        bbox_read['alpha']= bboxes[bbox]['alpha']
        bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
        bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
        bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
        bbox_read['right']= bboxes[bbox]['2d_bbox'][3]
        bbox_read['center'] =  np.array(bboxes[bbox]['center'])
        bbox_read['size'] =  np.array(bboxes[bbox]['size'])
        angle = bboxes[bbox]['rot_angle']
        axis = np.array(bboxes[bbox]['axis'])
        bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle) 
        boxes.append(bbox_read)

    return boxes 

def trans_obj_lidar_to_world(obj_rect, T_lidar2world):
    """
    Transform tracklet object points from lidar frame to world frame
    """
    # Transform 3d bbox points from lidar frame to world frame
    obj_rect = np.row_stack((obj_rect, np.ones(obj_rect.shape[1]))) # reshape to 4xn
    
    return np.dot(T_lidar2world, obj_rect)[:3, :]

def load_tracklets(xml_path:str, frame_ids:Union[List, None]= None)->tuple:
    """Loads dataset labels also referred to as tracklets, saving them individually for each frame.
    Args:
        xml_path (str): Path to the tracklets XML.
        frame_ids (Union[List, None], optional): Name of frame, eg: [0,1,3,4 ...]. Defaults to None -> all.
    Returns:
        tuple: Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. 
        First array contains coordinates of bounding box vertices for each object in the frame, and 
        the second array contains objects types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)
    
    # initialize dict with values as empty list
    frame_tracklets = defaultdict(lambda: []) 
    frame_tracklets_types = defaultdict(lambda: [])

    # loop over all data in tracklet
    for i, tracklet in enumerate(tracklets):

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])

        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            
            # frame in filtered frame
            if frame_ids is not None and absoluteFrameNumber not in frame_ids: 
                continue
            # determine if object is in the image and 
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            
            # Append data
            frame_tracklets[absoluteFrameNumber].append(cornerPosInVelo)
            frame_tracklets_types[absoluteFrameNumber].append(tracklet.objectType)

    return (frame_tracklets, frame_tracklets_types)