import os
import copy
import time
from datetime import datetime
import numpy as np
import open3d as o3d
import cv2
import pandas as pd
import yaml
import json
from scipy.spatial.transform import Rotation
import warnings
import open3d as o3d
import math 
from typing import Literal, Union, List
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt

def bbox2d_area(pos_xywh:List[float]) -> float:
    """Calculate the area of Bounding Box

    Args:
        pos_xyhw (List[float]): Coordinates of 2D bounding boxes in x, y, width, height format

    Returns:
        float: Area of bounding box
    """
    cx, cy, h, w =  pos_xywh
    return h*w

def xywh_to_xyxy(box):
    """
    Convert xywh format (center x, center y, width, height) to (x_min, y_min, x_max, y_max).

    Args:
        box (list or array): Bounding box in YOLO format [center_x, center_y, width, height]

    Returns:
        list: Bounding box in (x_min, y_min, x_max, y_max) format
    """
    center_x, center_y, width, height = box
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    x_max = center_x + (width / 2)
    y_max = center_y + (height / 2)
    
    return [x_min, y_min, x_max, y_max]

def xyxy_to_xywh(box):
    """
    Convert (x_min, y_min, x_max, y_max) format to xywh format (center x, center y, width, height).

    Args:
        box (list or array): Bounding box in (x_min, y_min, x_max, y_max) format

    Returns:
        list: Bounding box in YOLO format [center_x, center_y, width, height]
    """
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [center_x, center_y, width, height]

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # Calculate the area of intersection rectangle
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the area of both the prediction and ground-truth rectangles
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    # Calculate the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def base64_to_rgb(img_str:str) -> np.ndarray:
    """Covert base64 image to np.array

    Args:
        img_str (str): Image data in string

    Returns:
        np.ndarray: Image rgb format
    """
    decoded_value = base64.b64decode(img_str)
    image = np.frombuffer(decoded_value, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR) # Decode the image array to get the actual image
    return image

def np_to_base64(img_np:np.ndarray)->str:
    """Convert np image to base64 string

    Args:
        img_np (np.ndarray): Image array

    Returns:
        str: Image in base64 string format
    """
    _, buffer = cv2.imencode('.jpg', img_np)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    '''
    mydict = {'val':'it works'}
    nested_dict = {'val':'nested works too'}
    mydict = dotdict(mydict)
    mydict.val
    # 'it works'

    mydict.nested = dotdict(nested_dict)
    mydict.nested.val
    # 'nested works too'
    '''

def load_yaml(file_path:str):
    '''
    Load yaml file using file path
    :param file_path:str, path of yaml file
    :return:dict, serialised data in the form of dict
    '''
    # Load box24 yaml file
    with open(file_path, "r") as stream:
        try:
            data_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data_yaml

def load_json(file_path:str):
    '''
    Load Json file using path
    :param file_path:str, file path
    :return:dict, json file (dict)
    '''
    with open(file_path) as f:
        f_out = json.load(f)
    return f_out

def read_lines(file_path):
    """
    Read .txt file as string list
    Each lines treated as list containing string. 

    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def write_lines(file_path, data):
    """
    Write the data as string in txt file

    """
    with open(file_path, 'w') as f:
        lines = f.writelines(data)
        f.close()

def save_array_to_txt(data, save_path:str):
    """
    Save data to .txt file
    Args:
        data(list/ndarray): Array data
        save_path (str): path to save file
    """
    np.savetxt(save_path, data, delimiter=',', fmt="%f")

def save_json(data:dict, save_path:str ='', file_name:str='data'):
    '''
    Save dict to json file
    :param data:dict, input_data
    :param save_path:str, path to save json file
    :param file_name:str, file name
    :return: save .json file at provided path
    '''
    def NumpyEncoder(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))
    file_name = file_name + '_' + str(int(time.time())) + '.json'
    with open(os.path.join(save_path, file_name), 'w') as fp:
        json.dump(data, fp, default=NumpyEncoder, indent=4)

def load_csv(csv_path:str):
    '''
    Load csv file from path
    :param csv_path:str, path to csv file
    :return:
    '''

    # Read csv file
    df_input = pd.read_csv(csv_path, index_col=False)

    # Drop empty rows of df
    # df_input.drop(df_input[(df_input.Bbox_3d_id == '[nan]') & (df_input.Bbox_2d == '[nan]') & (df_input.pcd_3d == '[nan]') & (df_input.cam == '[nan]')].index, inplace=True)
    df_input.drop(df_input[df_input.pcd_3d == '[]'].index, inplace=True)
    # df_input.dropna(inplace=True)

    # Convert df dtype from string to list
    import ast
    # Change the df dtype from string to list
    input_cols = ['Bbox_3d_id', 'Bbox_3d', 'Bbox_2d', 'pcd_3d', 'cam', 'gnss', 'box_calib']
    for col in input_cols:
        df_input[col] = df_input[col].apply(str)  # convert to string to prevent eval error
        df_input[col] = df_input[col].apply(ast.literal_eval)  # convert string items to true dtype

    return df_input

def get_ego_speed(data_jsons: dict, seq_time:float= 3.0) -> float:
    """Calculate the average speed of the ego vehicle using the distance travelled by the ego vehicle

    Args:
        - data_jsons (dict): sequence data in deepen format
        - seq_time (float): sequence duration

    Returns:
        - float: ego average speed
    """

    ego_poses = list()
    for vals in data_jsons.values():
        position = list(vals['device_position'].values())
        ego_poses.append(position)
    ego_poses = np.array(ego_poses)
    delta_dist_poses = np.diff(ego_poses, axis=0)
    total_distance = np.sum(np.linalg.norm(delta_dist_poses, axis=1))
    ego_speed = total_distance/seq_time

    return ego_speed

def filter_data_outliers(data: list, std_limit:tuple=(-3,3)):
    '''
    Filter data for outliers using std (sigma) limit
    +-1 std from the Mean: 68%
    +-2 std from the Mean: 95%
    +-3 std from the Mean: 99.7%
    :param data:list, For data to be filtered
    :param std_limit:tuple, For std limit
    :return:
    list, filtered data
    list, outliers data
    '''
    # calculate summary statistics
    data_mean, data_std = np.nanmean(data), np.nanstd(data)
    # identify outliers
    #cut_off = data_std * 2
    l_limit = data_mean + data_std * std_limit[0]
    u_limit = data_mean + data_std * std_limit[1]

    # identify outliers
    # data_outliers = [x for x in data if x < l_limit or x > u_limit]

    # remove outliers
    # data_wo_outliers = [x for x in data if x >= l_limit and x <= u_limit]
    data_wo_outliers = []
    index = []
    for i, item in enumerate(data):
        if item >= l_limit and item <= u_limit:
            data_wo_outliers.append(item)
            index.append(i)

    return data_wo_outliers, index

def euler_to_quat(rot_xyz:list):
    """Convert euler angles (xyz) to quaternions angles

    Args:
        rot_xyz (list): Euler angles in radians (x,y,z) format

    Returns:
        list: quaternions angles (x, y, z, w) format
    """

    r = Rotation.from_euler('xyz', rot_xyz, degrees=False)
    return r.as_quat()

def quat_to_euler(rot_xyzw:list):
    """ Convert quaternions to euler angles

    Args:
        rot_xyzw (list): quaternions angles (x, y, z, w) format

    Returns:
        list: euler angles (xyz) in radians
    """
    r = Rotation.from_quat(rot_xyzw)
    return r.as_euler('xyz', degrees=False)

def create_pcd_tensor(points:list):
    """
    create open3d pcd object using xyz points and reflectance
    Args:
        points (list/ndarray): point coordinates in xyz format (n ,4)
    Return:
        pcd (open3d object): point cloud created using open3d tensor
    """
    pcd = o3d.t.geometry.PointCloud(points[:,:3]) # pcd tensor
    if np.shape(points)[1] == 4:
        pcd.point.intensity = points[:,3] # set extra attribute (reflectance)
    
    return pcd

def save_pcd(pcd, file_path):
    """
    Save pcd object to .pcd file using opnen3d pcd tensor object
    Intensity will save saved !!!
    """
    o3d.io.write_point_cloud(file_path, pcd.to_legacy(), write_ascii=True)
    warnings.warn("Saving Intensity is supported in open3d !!")

def rotx(t):
    """
    Rotation about the x-axis.
    args:
        t(float): rotation angle in radians
    return:
        rotation matrix (3x3)
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    """
    Rotation about the y-axis.
    args:
        t(float): rotation angle in radians
    return:
        rotation matrix (3x3)
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    """
    Rotation about the z-axis.
    args:
        t(float): rotation angle in radians
    return:
        rotation matrix (3x3)
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def gps_to_cartesian(lat=None,lon=None):
    """
    Converting lat/long to cartesian
    Args:
        lat (float): lattitude in deg
        lon (float): longitude in deg
    Return:
        xyz (list): cartesian coordinates in 'm'
    """

    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6378000 # radius of the earth in m
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return [x,y,z]

def Rt_to_T(R, t):
    """
    Transforation matrix from rotation matrix and translation vector.
    """

    R = R.reshape(3, 3) if R.ndim == 1 else R 
    t = t.reshape(3, 1) if t.ndim == 1 else t 
    T_mat = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
    
    return T_mat

def Rt_from_gps(lat, lon, roll, pitch, scale, alt=0.0, yaw=None):
    """
    Get Transformation matrix using latitude, longitude, altitude, roll angle, pitch angle, yaw angle and scale
    Args:
        lat(float): latitude in degrees
        lon(float): longitude in degrees
        roll(float): roll angle in radians
        pitch(float): pitch angle in radians
        scale(float): scale factor, calculated using latitude
        alt(float): antitude in meters
        yaw(float): yaw angle in radians
    Return:
        R(array): Rotation angles
        t(arraw): translation values
    """
    er = 6378137.0  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90. + lat) * np.pi / 360.0))
    tz = alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    
    if yaw:
        Rz = rotz(yaw)
        R = Rz.dot(Ry.dot(Rx))
    else:
        R = Ry.dot(Rx)

    # Combine the translation and rotation into a homogeneous transform
    return R, t

def create_3dbbox_mesh(corners_bbox, lines_boxes=None, line_color=[1, 0, 0]):
    """Create a 3dbbox meshes using corner points

    Args:
        corners_bbox (list): list of corner points of bounding boxes [3,8]
        lines_boxes (list): list of lines of bounding boxes
        line_color (list): line colour in RGB format
    Returns:
        list: list of 3dbbox meshes
    """

    # Create a 3D bounding box with 8 corner points
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    if lines_boxes is None:
        lines_boxes = [[[0, 1], [1, 2], [2, 3], [0, 3],
                [4, 5], [5, 6], [6, 7], [4, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]]] * len(corners_bbox)

    # Use the same color for all lines
    colors = [line_color for _ in range(len(lines_boxes[0]))]

    bbox_meshes =[]
    for bbox, lines_box in zip(corners_bbox, lines_boxes):
        bbox_mesh = o3d.geometry.LineSet()
        bbox_mesh.points = o3d.utility.Vector3dVector(bbox.T) # reshape to 8x3
        bbox_mesh.lines = o3d.utility.Vector2iVector(lines_box)
        bbox_mesh.colors = o3d.utility.Vector3dVector(colors)
        bbox_meshes.append(bbox_mesh)
    return bbox_meshes

def add_geometry_o3d(vis, point_cloud_o3d=None, bbox_meshes=None):

    # Add point cloud geometry:
    if point_cloud_o3d:
        vis.add_geometry(point_cloud_o3d)
    # Add mesh geometry:
    if bbox_meshes:
        for bbox_mesh in bbox_meshes:
            vis.add_geometry(bbox_mesh, reset_bounding_box=False)

def update_geometry_o3d(vis, point_cloud_o3d=None, bbox_meshes=None):
    """
    update open3d visualization geometery
    """
    # update point cloud geometry:
    if point_cloud_o3d:
        vis.update_geometry(point_cloud_o3d)
    # update mesh geometry:
    if bbox_meshes:
        for bbox_mesh in bbox_meshes:
            vis.update_geometry(bbox_mesh)

def remove_geometry(vis, meshes):
    """
    Remove the mesh geometry from o3d visualizer
    """
    if meshes:
        for mesh in meshes:
            vis.remove_geometry(mesh, reset_bounding_box=False)
            # vis.clear_geometries(mesh, reset_bounding_box=False)

def create_point_cloud_o3d(pcd_vehicle:np.ndarray, point_cloud_o3d:o3d.geometry=None, intensity_color:Union[str, list, None]=None):
    """ Create point open3d cloud object using point cloud points  in format (nx4)

    Args:
        - pcd_vehicle (ndarray): point cloud data in [nx4]
        - point_cloud_o3d (open3d geometry class, optional): geometry class of open3d. Defaults to None.
        - intensity_color (str, list, optional): colour value in [R, G, B], ['viridis', 'jet', 'gray', 'rainbow']. Defaults to None.

    Returns:
        - point_cloud_o3d (open3d geometry class): point cloud  geometry object
    """
    if not point_cloud_o3d:
         point_cloud_o3d = o3d.geometry.PointCloud() #create pcd object
    
    # add point to object
    point_cloud_o3d.points = o3d.utility.Vector3dVector(pcd_vehicle[:,:3])

    # Set the intensity values as colors for the point cloud
    if intensity_color is None:
        #1 Normalize the reflectance b/w 0 & 1
        reflectance = (pcd_vehicle[:,-1]-np.min(pcd_vehicle[:,-1]))/(np.max(pcd_vehicle[:,-1])-np.min(pcd_vehicle[:,-1]))
        
        #2 Create colours using normalized value
        colours = np.stack([reflectance, reflectance, reflectance], axis=1)
    elif isinstance(intensity_color, str):
        assert intensity_color in ['viridis', 'jet', 'gray', 'rainbow'], "colour should be from 'viridis', 'jet', 'gray', 'rainbow'"
        
        # Normalize the reflectance b/w 0 & 1
        reflectance = (pcd_vehicle[:, 3]-np.min(pcd_vehicle[:, 3]))/(np.max(pcd_vehicle[:, 3])-np.min(pcd_vehicle[:, 3]))
        colormap = plt.get_cmap(intensity_color) 
        colours = colormap(reflectance.flatten())[:, :3]  # Extract RGB values (n, 3)
    
    elif isinstance(intensity_color, list): #TODO: custom colour is not working
        assert len(intensity_color) == 3, "colour should be in format [r,g,b]"
        colours = np.repeat(intensity_color, len(pcd_vehicle)).reshape(-1, 3)
        colours = np.uint8(colours * 255)  # Convert to 0-255 range 
    
    else:
        raise ValueError("Unknown class of intensity colour")
    
    #3 Set reflectance as colour attribute
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colours)

    return point_cloud_o3d

def custom_visualization_o3d(
        vis:o3d.visualization, 
        pcd_size:float=None, 
        background_colour:Union[list, np.array]= None
        ) -> None:
    """
    Customize visualization window of open3d
    args:
        vis: o3d visualizer object
        pcd_size (float): size of point cloud
        background_colour (array): array for RGB value
    """
    # Custom open3d visualization options
    opt = vis.get_render_option()
    # change point cloud size
    if pcd_size:
        opt.point_size = pcd_size
    # change background colour to black
    if background_colour:
        opt.background_color = np.asarray(background_colour) 

def pcd_transformation(pcd:np.ndarray, T_mat:np.ndarray)-> np.ndarray:
    """
    Transform pcd (xyzi..) from source to target frame
    Args:
        pcd(np.array): point cloud array in shape nx4
        T_mat(np.array): Transformation array in shape 4x4
    Return:
        xyzi_target(np.ndarray): transformed point cloud to target frame, shape nx4
    """
    assert T_mat.shape == (4,4), "Shape of T_mat should be (4,4)"

    xyz_4xn = np.column_stack((pcd[:, :3] , np.ones(pcd.shape[0]))).T  # 4xn
    xyz_target = np.dot(T_mat, xyz_4xn)[:3, :].T # Transform to target frame, nx3
    xyz_target /= xyz_target[:, 2:3] # Normalize by depth, nx3
    xyzi_target = np.column_stack((xyz_target[:, :3], pcd[:, 3:])) # put back intensity and other attributes
    return xyzi_target

def get_bbox3d_corner_points(position, size, rotation_quaternion):
    """Create 3d bounding boxes corner points using position, size and orientation

    args:
        position(list): position of 3d bbox in format (x, y, z)
        size(list): size of 3d bbox in format (l, w, h)
        rotation_quaternion(list): quaternion in format (w, x, y, z)
    return:
        corner_points(array): eight corner points of 3d bbox
    """
    length, width, height = size
    #1: Convert quaternion to rotation matrix
    rotation_matrix = Quaternion(rotation_quaternion).rotation_matrix # (w, x, y, z) format
    # rotation_matrix = Rotation.from_quat(rotation_quaternion).as_matrix() #  (x, y, z, w) format

    #2: Calculate half dimensions
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2

    #3: Calculate corner offsets
    corner_offsets = [
        np.array([half_length, half_width, half_height]),
        np.array([-half_length, half_width, half_height]),
        np.array([-half_length, -half_width, half_height]),
        np.array([half_length, -half_width, half_height]),
        np.array([half_length, half_width, -half_height]),
        np.array([-half_length, half_width, -half_height]),
        np.array([-half_length, -half_width, -half_height]),
        np.array([half_length, -half_width, -half_height])
    ]
    
    #4: rotate offset values using rotation matrix
    rotated_corner_offsets = [np.dot(rotation_matrix, offset) for offset in corner_offsets]
    
    #5: Calculate coordinates of corner points using position and offsets 
    corner_points = [position + offset for offset in rotated_corner_offsets]

    return corner_points

def check_in_hull(point, hull):
    """Check if pcd points (`p`) in convex hull or not
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation will be computed.

    Args:
        point (array): pcd points
        hull (array | hull): object or np.array, hull object or 'MxK' array

    Returns:
        list : pcd belongs to hull
        list : hull mask
    """
    
    # Get hull object
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    
    # compute hull mask
    inhull_mask = hull.find_simplex(point[:, :3]) >= 0

    return point[inhull_mask], inhull_mask

def capture_screen(vis, image_path, upsample: int=1):
    """Save screen capture to image file

    Args:
        vis (open3d.visualizer): Visualizer class of open3d
        image_path (str): path to save image
        upsample (int): upsample times
    """
    # vis.capture_screen_image(os.path.join(args.data_io_path, 'pcd_bbox_vis_' + str(frame) + '.png'))
    image = vis.capture_screen_float_buffer(do_render=False)
    image = np.asarray(image)*255
    # Upsample the image
    image = cv2.resize(image, 
                       (image.shape[1]*upsample, image.shape[0]*upsample)
                       )
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    # Save image at path
    cv2.imwrite(image_path, image)

class NpEncoder(json.JSONEncoder):
    """convert nested numpy to nested list

    Args:
        json (json class): Extensible JSON <http://json.org> encoder for Python data structures.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return super(NpEncoder, self).default(obj)
    
def vertices_2_LWH(vertices:np.ndarray) -> np.ndarray:
    """Extract length, width and height from 3D bounding box vertices in world frame (front, left, up)
    """
    # Minimum and maximum X, Y, and Z coordinates
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])

    # Calculate length, width, and height
    length = max_x - min_x
    width = max_y - min_y
    height = max_z - min_z        
    return np.array([length, width, height])

def read_pcd(file_path:str) -> np.ndarray:
    """Read the pcd file based on type of file format [.bin, .txt, .npy]"""

    if '.bin' in os.path.basename(file_path):
        pcd = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)  # hardcode read .bin file, xyzi
    elif '.txt' in os.path.basename(file_path):
        pcd = np.loadtxt(file_path) # read txt file
    elif '.npy' in os.path.basename(file_path):
        pcd = np.load(file_path).astype(np.float32) # read .npy file
    else:
        raise "Unsupported pcd format"
    return pcd