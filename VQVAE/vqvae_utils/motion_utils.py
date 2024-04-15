import os
import sys

sys.path.append(os.path.dirname(__file__))

from pymo_vqvae.parsers import BVHParser
from pymo_vqvae.preprocessing import *
from pymo_vqvae.data import MocapData
from pymo_vqvae.writers import BVHWriter
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from typing import List
import numpy as np

FPS: int = 30

# added root, upleg, foot from 54 to 71 joints + 3 body_world position translations
joints : List[str] = ["root", "spine0", "spine1", "spine2", "spine3", "neck0", "head", "shoulder", "arm", "forearm", "wrist",
                      "upleg", "leg", "foot"]

# usefull if you want to use DiffuseStyleGesture rotmat representation
bone_names = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_l_foot_twist', 'b_l_foot', 'b_r_upleg',
              'b_r_leg', 'b_r_foot_twist', 'b_r_foot', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3',
              'b_neck0', 'b_head', 'b_l_shoulder', 'p_l_scap', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm',
              'b_l_wrist_twist', 'b_l_wrist', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3',
              'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3',
              'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3',
              'b_r_shoulder', 'p_r_scap', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist',
              'b_r_wrist', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3',
              'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3',
              'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3']


def load_bvh_file(file_path: str) -> MocapData:
    """
    Loads a BVH file.

    Args:
        file_path (str): Path to the BVH file.

    Returns:
        MocapData: Parsed motion capture data.
    """
    parser = BVHParser()
    parsed_data = parser.parse(file_path)
    return parsed_data


def create_data_pipeline() -> Pipeline:
    """
    Creates a pipeline for processing motion capture data.

    Returns:
        Pipeline: The data processing pipeline.
    """
    return Pipeline([
        ('root', RootTransformer('hip_centric')),
        ('exp', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover()),
        ('jtsel', JointSelector(joints, include_root=True)),
        ('np', Numpyfier())
    ])


def create_TAG2G_pipeline() -> Pipeline:
    """
    Creates a pipeline for processing motion capture data.

    Returns:
        Pipeline: The data processing pipeline.
    """
    pipeline = Pipeline([
        ('exp', TAG2G_MocapParameterizer(param_type='TAG2G_expmap', bworld_translation_scaler=100)),
        ('cnst', ConstantsRemover()),
        ('jtsel', TAG2G_JointSelector(joints, include_body_world=True)),
        ('np', Numpyfier()),
    ])
    return pipeline

def create_TWH_pipeline() -> Pipeline:

    # rotmat implementation

    return Pipeline([
        ('jtsel', JointSelector(bone_names, include_root=False)),
        ('np', Numpyfier()),
    ])


def process_pipeline(parsed_data: MocapData, pipeline: Pipeline) -> np.ndarray:
    """
    Processes motion capture data using a pipeline.

    Args:
        parsed_data (MocapData): The parsed motion capture data.
        pipeline (Pipeline): The data processing pipeline.

    Returns:
        np.ndarray: Processed data samples.
    """

    processed_samples = pipeline.fit_transform([parsed_data])
    return processed_samples[0]


def inverse_process_pipeline(gesture_data: np.ndarray, pipeline: Pipeline) -> MocapData:

    mocap_data = pipeline.inverse_transform([gesture_data])
    return mocap_data[0]


def process_TWH_pipeline(parsed_data: MocapData, pipeline : Pipeline) -> np.ndarray:

    out_data = pipeline.fit_transform([parsed_data])
    # euler -> rotatiojn matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6))  # 3 pos (XYZ), 3 rot (ZXY)
    out_matrix = np.zeros(
        (out_data.shape[0], out_data.shape[1], out_data.shape[2], 12))  # 3 pos, 1 rot matrix (9 elements)
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            for k in range(out_data.shape[2]):  # joints
                out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0]


def inverse_process_TWH_pipeline(predicted_gesture, pipeline : Pipeline) -> MocapData:

    # implemented following DiffuseStyleGesture/process_TWH_bvh.py - pose2bvh

    # smoothing
    n_poses = predicted_gesture.shape[0]
    out_poses = np.zeros((n_poses, predicted_gesture.shape[1]))
    for i in range(predicted_gesture.shape[1]):
        out_poses[:, i] = savgol_filter(predicted_gesture[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal

    # rotation matrix to euler angles
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 12))  # (n_frames, n_joints, 12)
    out_data = np.zeros((out_poses.shape[0], out_poses.shape[1], 6))
    for i in range(out_poses.shape[0]):  # frames
        for j in range(out_poses.shape[1]):  # joints
            out_data[i, j, :3] = out_poses[i, j, :3]
            r = R.from_matrix(out_poses[i, j, 3:].reshape(3, 3))
            out_data[i, j, 3:] = r.as_euler('ZXY', degrees=True).flatten()

    out_data = out_data.reshape(out_data.shape[0], -1)
    bvh_data = pipeline.inverse_transform([out_data])[0]

    return bvh_data


def split_bvh_into_blocks(processed_data: np.ndarray, beats: np.ndarray) -> List[np.ndarray]:
    """
    Splits BVH data into blocks based on beat intervals.

    Args:
        processed_data (np.ndarray): The processed motion capture data.
        beats (np.ndarray): List of beat frame indices.

    Returns:
        List[np.ndarray]: BVH data blocks.
    """

    blocks = []
    frame_beats = beats

    for i in range(len(frame_beats) - 1):
        start = frame_beats[i]
        end = frame_beats[i + 1]

        assert start != end, "Start and end frame indices must be different"

        block = processed_data[start:end]
        blocks.append(block)

    return blocks


def gesture_smoothing(sample):

    # gesture smoothing
    gesture = np.zeros((sample.shape[0], sample.shape[1]))
    for i in range(sample.shape[1]):
        gesture[:, i] = savgol_filter(sample[:, i], 15, 2)

    return gesture


def save_gesture_as_bvh(mocap_data:MocapData, file_path:str, file_name:str):

    assert os.path.isdir(file_path), f"Provided file path is not an existing directory"

    # instancing writer and write to a bvh
    writer = BVHWriter()

    dump_file = os.path.join(file_path, f"{file_name}.bvh")
    print(f"Saving motion to '{dump_file}'")
    with open(dump_file, "w") as file:
        writer.write(X=mocap_data,
                     ofile=file,
                     framerate=30)


if __name__ == "__main__":

    # testing load and gesture process pipeline: 21-12 TESTED OK

    bvh_path = r"C:\Users\faval\genea2023_dataset\trn\main-agent\bvh"
    bvh_path = os.path.join(bvh_path, os.listdir(bvh_path)[0])
    assert os.path.isfile(bvh_path), "Provided path is not a file"

    pipeline = create_TWH_pipeline()

    bvh_data = load_bvh_file(bvh_path)
    gesture_data = process_TWH_pipeline(parsed_data=bvh_data,
                                        pipeline=pipeline)

    print(f"Gesture data shape: {gesture_data.shape}")

    bvh_data = inverse_process_TWH_pipeline(predicted_gesture=gesture_data,
                                            pipeline=pipeline)

    print(type(bvh_data))



