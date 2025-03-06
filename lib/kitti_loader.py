import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../.."))) #add package path to python path
import numpy as np
from pathlib import Path
import glob, tqdm
import matplotlib.pyplot as plt
from typing import Literal, Union, List, ClassVar
from pydantic import BaseModel, Field
import open3d as o3d
from lib import tools
from lib import utils_kitti as utils

class KITTIConfigs(BaseModel):
    """Configuration of KITTI dataset"""
    CAMERAS: ClassVar[list[str]] = ['00', '01', '02', '03']

class KITTILoader:
    def __init__(self, kitti_root:str, seq_name:str) -> None:
        """Constructor for KITTI Loader

        Args:
            kitti_root (str): Root path of KITTI dataset (eg: .../kitti/2011_09_26 )
            seq_name (str): Name of the sequence (eg: '0009')
        """
        # Root directory.
        self.kitti_root = Path(kitti_root)
        self.seq_name = seq_name
        if not self.kitti_root.is_dir():
            raise FileNotFoundError(f"KITTI {kitti_root} not found.")

        # Other directories.
        self.calibration_dir = self.kitti_root / f"{self.kitti_root.name}_calib"
        self.sensor_data_dir = self.kitti_root / f"{self.kitti_root.name}_drive_{self.seq_name}_sync"
        self.tracklets_dir = self.kitti_root / f"{self.kitti_root.name}_drive_{self.seq_name}_tracklets"

        # Check if all directories exist.
        if not self.calibration_dir.is_dir():
            raise FileNotFoundError(
                f"Calibration dir {self.calibration_dir} not found."
            )
        if not self.sensor_data_dir.is_dir():
            raise FileNotFoundError(f"Data poses dir {self.sensor_data_dir} not found.")
        
        if not self.tracklets_dir.is_dir():
            raise FileNotFoundError(
                f"Data 2D raw dir {self.data_2d_raw_dir} not found."
            )
    
    @staticmethod
    def extract_cam_calib(clib_path: str, cam_name: KITTIConfigs.CAMERAS) -> dict: # type: ignore
        """Get the camera calibration matrix
        Args:
            clib_path (str): Path of calibration file
            camera_name (KITTIConfigs.CAMERAS): Name of camera, ['00', '01', '02', '03']
        Returns:
            dict: Dictionary containing camera matrices, ['S', 'K', 'D', 'R', 't', 'S_rect', 'R_rect', 'P_rect']
        """
        # Check if cam_name is one of the allowed values
        assert cam_name in KITTIConfigs.CAMERAS, f"Invalid camera name: {cam_name}"

        # Read .text file
        with open(clib_path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]

        cam_dict  = {k: [] for k in ['S', 'K', 'D', 'R', 't', 'S_rect', 'R_rect', 'P_rect']}

        for line in lines:
            # Skew factor
            if f'S_{cam_name}' in line[0]: cam_dict['S'] = np.array([np.float32(i) for i in line[1:]])
            # Intrinsics matrix (3x3)
            if f'K_{cam_name}' in line[0]: cam_dict['K'] = np.reshape([np.float32(i) for i in line[1:]], (3,3)) 
            # Distortion coefficient
            if f'D_{cam_name}' in line[0]: cam_dict['D'] = np.array([np.float32(i) for i in line[1:]])    
            #Rotation matrix (3x3)
            if f'R_{cam_name}' in line[0]: cam_dict['R'] = np.reshape([np.float32(i) for i in line[1:]], (3,3)) 
            # Translation vector
            if f'T_{cam_name}' in line[0]: cam_dict['t'] = np.array([np.float32(i) for i in line[1:]])
             # Rectification skew factor
            if f'S_rect_{cam_name}' in line[0]: cam_dict['S_rect'] = np.array([np.float32(i) for i in line[1:]])
            # Rectification rotation matrix (3x3)
            if f'R_rect_{cam_name}' in line[0]: cam_dict['R_rect'] = np.reshape([np.float32(i) for i in line[1:]], (3,3))   
            # Rectified camera projection matrix (3x4)
            if f'P_rect_{cam_name}' in line[0]: cam_dict['P_rect'] = np.reshape([np.float32(i) for i in line[1:]], (3,4))   

        return cam_dict
    
    @staticmethod
    def extract_Rt(calib_path: str)->np.ndarray:
        """Extract Rt transformation matrix for imu and velo from calibration.txt
        Args:
            calib_path (str): Path of clibration file
        Returns:
            np.ndarray: T(Rt) matrix, shape (3,4)
        """

        # Read .text file
        with open(calib_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        R_data = np.asarray([np.float32(i) for i in lines[1].split()[1:]])
        t_data = np.asarray([np.float32(i) for i in lines[2].split()[1:]])

        # Calculate T matrix using R and t data
        T_mat = tools.Rt_to_T(R_data, t_data) # Rt = T

        return T_mat

    @staticmethod
    def calc_velo_to_cam_img(cam_calib_path:str, cam_velo_path:str, cam_name:KITTIConfigs.CAMERAS)->tuple: # type: ignore
        """Transformation matrix from Lidar to Camera and img frame \n
        Transformation path: velo -> cam_reference(cam0) -> cam_0/1/2/3 -> cam_rectification -> cam_projections (intrinsics)
        Args:
            cam_calib_path (str): Path of camera calibration file
            cam_velo_path (str): Path of velodyne calibration file
            camera_name (KITTIConfigs.CAMERAS): Name of camera, ['00', '01', '02', '03']
        Returns:
            tuple: Tuple containing T_velo_cam matrix (4, 4), T_velo_img matrix (3, 4)
        """
        # Load Camera Calibration Data        
        cam_mats = KITTILoader.extract_cam_calib(cam_calib_path, cam_name)

        # Load velo to camera reference T matrix
        T_velo_camref = KITTILoader.extract_Rt(cam_velo_path)

        # get rectified rotation matrices (camera --> rectified left camera)
        R_ref_rect = cam_mats['R_rect']

        # get rigid transformation from Cam0 (ref) to Camera
        R, t  = cam_mats['R'], cam_mats['t']

        # add (0,0,0) translation and convert to homogeneous coordinates (4x4)
        R_ref_rect = np.insert(R_ref_rect, 3, values=[0,0,0], axis=0)
        R_ref_rect = np.insert(R_ref_rect, 3, values=[0,0,0,1], axis=1)

        # get cam0 to camera rigid body transformation in homogeneous coordinates 
        T_camref_cam = tools.Rt_to_T(R, t)
        
        # transform from velo (LiDAR) to color camera (shape 3x4)
        T_velo_cam = R_ref_rect @ T_camref_cam @ T_velo_camref

        # get projection matrices (rectified camera --> Projection left camera (u,v,z))
        P_rect_cam = cam_mats['P_rect']

        # Finally projection matrix
        P_velo_img = P_rect_cam @ T_velo_cam

        return T_velo_cam, P_velo_img

    def load_images(self, camera_name:KITTIConfigs.CAMERAS, frame_ids:Union[List, None]= None): # type: ignore
        """Read the image as per camera name
        Args:
            camera_name (KITTIConfigs.CAMERAS): Name of camera, ['00', '01', '02', '03']
            frame_ids (Union[List, None], optional): Name of frame, eg: [0,1,3,4 ...]. Defaults to None->all.
        Returns:
            np.ndarray: RGB images, float32, shape [N, H, W, 3]
        """
        # Get image paths
        im_paths = self.get_image_paths(camera_name, frame_ids)

        # Read images
        imgs = []
        for im_path in tqdm.tqdm(im_paths, desc="Loading Images..."):
            imgs.append(plt.imread(im_path))
        # imgs = np.stack(imgs, axis=0)
        return imgs

    def get_image_paths(self, camera_name:KITTIConfigs.CAMERAS, frame_ids:Union[List, None]= None)->list: # type: ignore
        """Read the image paths as per camera name
        Args:
            camera_name (KITTIConfigs.CAMERAS): Name of camera, ['00', '01', '02', '03']
            frame_ids (Union[List, None], optional): Name of frame, eg: [0,1,3,4 ...]. Defaults to None->all.
        Returns:
            list: image paths.
        """
        # Get image paths.
        im_dir = self.sensor_data_dir / f"image_{camera_name}" / "data"
        if frame_ids is None:
            im_paths = sorted(glob.glob(str(im_dir/ '*.png')))
        else:
            im_paths = [str(im_dir / f"{frame_id:010d}.png") for frame_id in frame_ids]
        
        # Sanity check
        for im_path in im_paths:
            if not os.path.isfile(im_path):
                raise FileNotFoundError(f"Image {im_path} not found.")

        return im_paths
    
    def load_pcds(self, frame_ids:Union[List, None]= None):
        """ Load pcds files
        Args:
            frame_ids (Union[List, None], optional): List of frames, eg. [3,5,6,...]. Defaults to None->all.
        Returns:
            np.ndarray: Array of pcd, shape (N,4)
        """
        # Get pcd paths
        pcd_paths = self.get_pcd_paths(frame_ids) 

        # Read pcds
        pcds = []
        for pcd_path in tqdm.tqdm(pcd_paths, desc="Loading pcds..."):
            pcds.append(tools.read_pcd(pcd_path))
        # pcds = np.stack(pcds, axis=0)
        return pcds
    
    def get_pcd_paths(self, frame_ids:Union[List, None]= None)->list:
        """Read the velodyne pcd paths
        Args:
            frame_ids (Union[List, None], optional): List of frames, eg. [3,5,6,...]. Defaults to None->all.
        Returns:
            list: List of path of pcds
        """
        # Get image paths.
        velo_dir = self.sensor_data_dir / "velodyne_points" / "data"
        if frame_ids is None:
            pcd_paths = sorted(glob.glob(str(velo_dir/ '*.bin')))
        else:
            pcd_paths = [str(velo_dir / f"{frame_id:010d}.bin") for frame_id in frame_ids]
        
        # Sanity check
        for pcd_path in pcd_paths:
            if not os.path.isfile(pcd_path):
                raise FileNotFoundError(f"Image {pcd_path} not found.")

        return pcd_paths

    def get_gps_paths(self, frame_ids:Union[List, None]= None):
        gps_dir = self.sensor_data_dir / 'oxts/data'
        if frame_ids is None:
            gps_files = sorted(glob.glob(str(gps_dir/ '*.txt')))
        else:
            gps_files = [str(gps_dir / f"{frame_id:010d}.txt") for frame_id in frame_ids]
        return gps_files
    
    def get_poses_imu(self, frame_ids:Union[List, None]= None):
        """Transformation matrix of IMU to world frame
        Args:
            frame_ids (Union[List, None], optional): Name of frame, eg: [0,1,3,4 ...]. Defaults to None->all.
        Returns:
            np.ndarray: T matrix IMU to world coordinate frame, shape (N, 4, 4)
        """
        # Get paths
        gps_paths = self.get_gps_paths(frame_ids)

        # Get transformation matrices from vehicle to map for the frames
        T_mats_ego2map = np.array(utils.get_trans_imu_seq(gps_paths))

        return T_mats_ego2map
    
    def get_poses_velo(self, frame_ids:Union[List, None]= None)->np.ndarray:
        """Transformation matrix of velodyne LiDAR to world frame
        Args:
            frame_ids (Union[List, None], optional): Name of frame, eg: [0,1,3,4 ...]. Defaults to None->all.
        Returns:
            np.ndarray: T matrix velodyne to world coordinate frame, shape (N, 4, 4)
        """
        # Get velodyne frames time stamp values
        # timestamps_velo_dir = self.sensor_data_dir / "velodyne_points" / "timestamps.txt"
        # timestamps_velo =  utils.timestamp_to_float(str(timestamps_velo_dir))
        
        # Get gps/imu frames time stamp values
        # timestamps_imu = utils.timestamp_to_float(self.sensor_data_dir / 'oxts' / 'timestamps.txt')
        
        # Get gps/imu poses in world
        T_imu2world = self.get_poses_imu(frame_ids)

        # Calculate velodyne poses in world
        T_imu2velo = self.extract_Rt(self.calibration_dir/'calib_imu_to_velo.txt')
        T_velo2world = np.dot(T_imu2world, T_imu2velo)
        return T_velo2world
    
    def get_poses_cam(self, camera_name:KITTIConfigs.CAMERAS, frame_ids:Union[List, None]= None)-> np.ndarray: # type: ignore
        """Transformation matrix of camera to world frame
        Args:
            camera_name (KITTIConfigs.CAMERAS): Name of camera, ['00', '01', '02', '03']
            frame_ids (Union[List, None], optional): Name of frame, eg: [0,1,3,4 ...]. Defaults to None->all.
        Returns:
            np.ndarray: T matrix camera to world coordinate frame, shape (N, 4, 4)
        """
        # Get velodyne frames time stamp values
        # timestamps_cam_dir = self.sensor_data_dir / f"image_{camera_name}" / "timestamps.txt"
        # timestamps_cam =  utils.timestamp_to_float(str(timestamps_cam_dir))
        
        # Get gps/imu frames time stamp values
        # timestamps_imu = utils.timestamp_to_float(self.sensor_data_dir / 'oxts' / 'timestamps.txt')
        
        # Get gps/imu poses in world
        # T_imu2world = self.get_poses_imu(frame_ids)

        # Calculate velodyne poses in world map
        T_velo2world = self.get_poses_velo(frame_ids)

        # Calculate velo to cam pose T
        T_velo2cam, T_velo2img = KITTILoader.calc_velo_to_cam_img(
            self.calibration_dir/'calib_cam_to_cam.txt', 
            self.calibration_dir/'calib_velo_to_cam.txt',
            camera_name)
        
        # Finally camera to world transformation matrix
        T_cam2world = np.dot(T_velo2world, T_velo2cam)
        
        return T_cam2world
    
    def load_camera(self, camera_name: KITTIConfigs.CAMERAS)->tuple:  # type: ignore
        """Load rigid body transformations
        Args:
            camera_name (KITTIConfigs.CAMERAS): Name of camera from ['00', '01', '02', '03']
        Returns:
            np.ndarray: tuple of K , T_velo_cam, T_velo_img matrices
        """
        # Dirs  
        cam_clib_path = self.calibration_dir/'calib_cam_to_cam.txt'
        velo2cam_ref_clib_path = self.calibration_dir/'calib_velo_to_cam.txt'

        # Transformation matrix
        T_velo_cam, T_velo_img = KITTILoader.calc_velo_to_cam_img(cam_clib_path, velo2cam_ref_clib_path, camera_name)

        # Load camera transformation matrix
        cam_dict = KITTILoader.extract_cam_calib(cam_clib_path, camera_name)
        K = cam_dict['P_rect'][:3,:3]
        
        return (K , T_velo_cam, T_velo_img)
    
    def load_tracklets(self, frame_ids:Union[List, None]= None)->tuple:
        """Load the annotations
        Args:
            frame_ids (Union[List, None], optional): Name of frame, eg: [0,1,3,4 ...]. Defaults to None->.
        Returns:
            tuple: Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. 
            First array contains coordinates of bounding box vertices for each object in the frame, and 
            the second array contains objects types as strings.
        """
        tracklet_rects, tracklet_types = utils.load_tracklets(
            xml_path=str(self.tracklets_dir/'tracklet_labels.xml'),
            frame_ids=frame_ids
            )
        
        return (tracklet_rects, tracklet_types)
    
if __name__ == "__main__":

    # Params
    frame_ids = list(range(10,21))
    frame_id = 5
    camera_id = "02"

    # Initialize Kitti loader
    kitti = KITTILoader(
        kitti_root="/mnt/c/Users/Gaurav/Downloads/Datasets/kitti/2011_09_26",
        seq_name="0009"
    )
    poses_imu = kitti.get_poses_imu(frame_ids)
    poses_velo = kitti.get_poses_velo(frame_ids)
    poses_cam02 = kitti.get_poses_cam(camera_id, frame_ids)
    pcd_paths = kitti.get_pcd_paths(frame_ids)
    cam02_paths = kitti.get_image_paths(camera_id, frame_ids)
    K , T_velo_cam, T_velo_img = kitti.load_camera(camera_id)
    tracklet_rects, tracklet_types = kitti.load_tracklets()

    # Visualize image
    img = plt.imread(cam02_paths[frame_id])
    plt.imshow(img)
    
    # Visualize image and pcd fusion
    pcd_velo = tools.read_pcd(pcd_paths[frame_id]) # Lidar frame
    pcd_img = tools.pcd_transformation(pcd_velo, np.vstack((T_velo_img, [0,0,0,1])))
    plt.scatter(pcd_img[:,0], pcd_img[:,1], s=2, c=pcd_velo[:,3])
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)

    # Visualize point cloud
    pcd = tools.read_pcd(pcd_paths[frame_id]) # Lidar frame
    pcd_img = tools.pcd_transformation(pcd, T_velo_img)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    tools.custom_visualization_o3d(vis, pcd_size=1.0)

    point_cloud_o3d = o3d.geometry.PointCloud()   
    point_cloud_o3d = tools.create_point_cloud_o3d(pcd, point_cloud_o3d)
    tools.add_geometry_o3d(vis, point_cloud_o3d)  
    
    vis.poll_events()
    vis.update_renderer()    