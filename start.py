import os
import time
import cv2
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import align.detect_face as detect_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from src.sort import Sort

logger = Logger()
video_index = 1  

def main_webcam():
    global colours, img_size
    global video_index

    videos_dir = "videos"
    mkdir(videos_dir)

    detect_interval = 1
    margin = 10
    show_rate = 1
    face_score_threshold = 0.85

    tracker = Sort()

    writer = None       
    frame_count = 0
    fps = 20
    post_event_duration = 5  
    last_seen_time = None   
    start_time = None         
    frame_h, frame_w = 0, 0

    logger.info('Start webcam tracking...')
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Erro: não consegui abrir a webcam")
        return

    ret, test_frame = cam.read()
    if not ret:
        print("Erro: não consegui capturar frame inicial")
        return
    frame_h, frame_w = test_frame.shape[:2]

    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                              log_device_placement=False)) as sess:
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))

            minsize = 40
            threshold = [0.6, 0.7, 0.7]
            factor = 0.709

            frame_idx = 0
            while True:
                ret, frame = cam.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (frame_w, frame_h))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                final_faces = []

                if frame_idx % detect_interval == 0:
                    img_size = np.asarray(frame.shape)[0:2]
                    faces, _ = detect_face.detect_face(rgb_frame, minsize, pnet, rnet, onet, threshold, factor)
                    if faces.shape[0] > 0:
                        for i, f in enumerate(faces):
                            score = round(f[4], 6)
                            if score > face_score_threshold:
                                det = np.squeeze(f[0:4])
                                det[0] = max(det[0] - margin, 0)
                                det[1] = max(det[1] - margin, 0)
                                det[2] = min(det[2] + margin, img_size[1])
                                det[3] = min(det[3] + margin, img_size[0])
                                final_faces.append(f)

                trackers_data = tracker.update(np.array(final_faces), img_size, None, [], detect_interval)
                frame_idx += 1
                current_time = time.time()

                if len(trackers_data) > 0:
                    last_seen_time = current_time
                    if writer is None:

                        start_time = current_time
                        start_str = time.strftime("%Y%m%d-%H%M%S - ", time.localtime(current_time))
                        out_path = os.path.join(videos_dir, f"{video_index} - {start_str}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
                        frame_count = 0
                        logger.info(f"Evento detectado - iniciando gravação -> {out_path}")
                        
                        video_index += 1

                if writer is not None:

                    writer.write(frame)
                    frame_count += 1

                    if last_seen_time and (current_time - last_seen_time > post_event_duration):
                        writer.release()
                        logger.info(f"Gravação finalizada com {frame_count} frames")
                        writer = None
                        last_seen_time = None
                        frame_count = 0

                for d in trackers_data:
                    d = d.astype(np.int32)
                    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {int(d[4])}", (d[0], d[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                frame_show = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
                cv2.imshow("Frame", frame_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if writer is not None:
        writer.release()
        logger.info(f"Fechado gravador aberto ao encerrar programa")

    cam.release()
    cv2.destroyAllWindows()



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.', default='videos')
    parser.add_argument('--output_path', type=str,
                        help='Path to save face',
                        default='facepics')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int, default=1)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int, default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float, default=0.7)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float, default=1)
    parser.add_argument('--face_score_threshold',
                        help='The threshold of the extracted faces,range 0<x<=1',
                        type=float, default=0.85)
    parser.add_argument('--face_landmarks',
                        help='Draw five face landmarks on extracted face or not ', action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main_webcam()

