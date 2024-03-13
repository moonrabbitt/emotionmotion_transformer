import cv2
import numpy as np
import os
import json
from multiprocessing import Pool, cpu_count
import glob
import torch
from tqdm import tqdm

def load_model():
    protoFile = "models/openpose_25/pose_deploy.prototxt"
    weightsFile = "models/openpose_25/pose_iter_584000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    if device == 'cpu':
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    elif device == 'cuda':
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

def draw_pose(frame, keypoints, limb_connections, keypoints_dict):
    for connection in limb_connections:
        partA, partB = keypoints_dict[connection[0]], keypoints_dict[connection[1]]
        if keypoints[partA] and keypoints[partB]:
            cv2.line(frame, keypoints[partA], keypoints[partB], (0, 255, 255), 2)
            cv2.circle(frame, keypoints[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, keypoints[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    return frame

def get_pose_keypoints(frame, net, nPoints, threshold):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    inHeight = 368
    inWidth = int((inHeight / frame_height) * frame_width)
    inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp_blob)
    output = net.forward()
    keypoints = [None] * nPoints
    for i in range(nPoints):
        prob_map = output[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(prob_map)
        x = (frame_width * point[0]) / output.shape[3]
        y = (frame_height * point[1]) / output.shape[2]
        if prob > threshold:
            keypoints[i] = (int(x), int(y))
    return keypoints

def process_video(file_path):
    net = load_model()
    nPoints = 25
    threshold = 0.1
    
    # Define BODY_25 Keypoints
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                        'L-Elb', 'L-Wr', 'MidHip', 'R-Hip', 'R-Knee', 'R-Ank', 
                        'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 
                        'L-Ear', 'L-BigToe', 'L-SmallToe', 'L-Heel', 'R-BigToe', 
                        'R-SmallToe', 'R-Heel']

    # Define limb connections
    limb_connections = [
        ("Nose", "Neck"),
        ("Neck", "R-Sho"),
        ("R-Sho", "R-Elb"),
        ("R-Elb", "R-Wr"),
        ("Neck", "L-Sho"),
        ("L-Sho", "L-Elb"),
        ("L-Elb", "L-Wr"),
        ("Neck", "MidHip"),
        ("MidHip", "R-Hip"),
        ("R-Hip", "R-Knee"),
        ("R-Knee", "R-Ank"),
        ("MidHip", "L-Hip"),
        ("L-Hip", "L-Knee"),
        ("L-Knee", "L-Ank"),
        ("Nose", "R-Eye"),
        ("R-Eye", "R-Ear"),
        ("Nose", "L-Eye"),
        ("L-Eye", "L-Ear"),
        ("L-Ank", "L-BigToe"),
        ("L-Ank", "L-SmallToe"),
        ("L-Ank", "L-Heel"),
        ("R-Ank", "R-BigToe"),
        ("R-Ank", "R-SmallToe"),
        ("R-Ank", "R-Heel")
    ]
    keypoints_dict = {key: idx for idx, key in enumerate(keypointsMapping)}
    
    output_video_path = os.path.join(os.path.dirname(file_path), "annotated_" + os.path.basename(file_path))
    keypoints_output_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split('.')[0] + "_keypoints.txt")
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {file_path}.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(file_path)}", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = get_pose_keypoints(frame, net, nPoints, threshold)
        keypoints_dict[frame_idx] = keypoints
        output_frame = draw_pose(frame.copy(), keypoints, limb_connections, keypoints_dict)
        out.write(output_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save keypoints to a text file
    with open(keypoints_output_path, 'w') as f:
        json.dump(keypoints_dict, f)

    print(f"Finished processing {file_path}. Results saved to {output_video_path} and {keypoints_output_path}")

if __name__ == '__main__':
    video_files = glob.glob("G:\\UAL_Thesis\\affective_computing_datasets\\DanceDBrenders\\DanceDB\\*\\*.mp4")
    num_workers = max(cpu_count() - 2, 1)
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_video, video_files), total=len(video_files), desc="Processing Videos"))

