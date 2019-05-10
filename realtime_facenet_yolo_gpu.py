from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from yolo.yolo import *
from svm_models import get_X_train, features_model

from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./align/yolo_weights/YOLO_Face.h5',
                    help='path to model weights file')
    parser.add_argument('--anchors', type=str, default='./align/yolo_cfg/yolo_anchors.txt',
                    help='path to anchor definitions')
    parser.add_argument('--classes', type=str, default='./align/yolo_cfg/face_classes.txt',
                    help='path to class definitions')
    parser.add_argument('--score', type=float, default=0.5,
                    help='the score threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                    help='the iou threshold')
    parser.add_argument('--img-size', type=list, action='store',
                    default=(416, 416), help='input image size')
    args = parser.parse_args()
    return args;

# print('Creating networks and loading parameters')

# Load YOLO V2 model.
# net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def _main():

    args = get_args()


    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # pnet, rnet, onet = detect_face.create_mtcnn(sess, './models/')

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160

            print('Loading feature extraction model')
            modeldir = './models/20180402-114759'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            ## Get age gender race models:
            X_train = get_X_train()
            gender_model = features_model(X_train, "gender")
            age_model = features_model(X_train, "age")
            race_model = features_model(X_train, "race")
            # print("Loading age model")
            # age_filename = './myclassifier/age_nn_classifier.pkl'
            # age_filename_exp = os.path.expanduser(age_filename)
            # with open(age_filename_exp, 'rb') as f: age_model = pickle.load(f)
            # print("Loading gender model")
            # gender_filename = './myclassifier/gender_nn_classifier.pkl'
            # gender_filename_exp = os.path.expanduser(gender_filename)
            # with open(gender_filename_exp, 'rb') as f: gender_model = pickle.load(f)
            # print("Loading race model")
            # race_filename = './myclassifier/race_nn_classifier.pkl'
            # race_filename_exp = os.path.expanduser(race_filename)
            # with open(race_filename_exp, 'rb') as f: race_model = pickle.load(f)

            classifier_filename = './myclassifier/my_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)

            video_capture = cv2.VideoCapture(0)
            # video_capture = cv2.VideoCapture('rtsp://10.89.220.212:7754/h264_pcm.sdp')
            c = 0

            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            fourcc = video_capture.get(cv2.CAP_PROP_FOURCC)
            codec = decode_fourcc(fourcc)
            print("Codec: " + codec)
            print('Start Recognition!')
            prevTime = 0
            myYolo = YOLO(args)
            while True:
                ret, frame = video_capture.read()
                frame = cv2.flip(frame, 1)
                # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

                curTime = time.time()    # calc fps
                timeF = frame_interval

                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    #prin                            print(age_preds[0])t(frame.shape[0])
                    #print(frame.shape[1])

                    image = Image.fromarray(frame)
                    img, bounding_boxes = myYolo.detect_image(image)

                    # Remove the bounding boxes with low confidence
                    nrof_faces = len(bounding_boxes)
                    ## Use MTCNN to get the bounding boxes
                    # bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    # nrof_faces = bounding_boxes.shape[0]
                    #print('Detected_FaceNum: %d' % nrof_faces)

                    if nrof_faces > 0:
                        # det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        # cropped = []
                        # scaled = []
                        # scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = bounding_boxes[i][0]
                            bb[i][1] = bounding_boxes[i][1]
                            bb[i][2] = bounding_boxes[i][2]
                            bb[i][3] = bounding_boxes[i][3]

                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('face is inner of range!')
                                continue

                            # cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            # cropped[0] = facenet.flip(cropped[0], False)image_
                            # scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                            # scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
                            #                        interpolation=cv2.INTER_CUBIC)
                            # scaled[0] = facenet.prewhiten(scaled[0])
                            # scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                            # feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}

                            cropped = (frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            print("{0} {1} {2} {3}".format(bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
                            cropped = facenet.flip(cropped, False)
                            scaled = (misc.imresize(cropped, (image_size, image_size), interp='bilinear'))
                            scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = (scaled.reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}

                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                            age_preds = age_model.predict(emb_array)
                            gender_preds = gender_model.predict(emb_array)
                            race_preds = race_model.predict(emb_array)

                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            print(best_class_probabilities)
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            age_str = "Age: " + str(age_preds.tolist()[0])
                            gender_dict = {0: "Male", 1: "Female"}
                            gender_str = "Gender: " + gender_dict[gender_preds.tolist()[0]]
                            race_dict = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
                            race_str = "Race: " + race_dict[race_preds.tolist()[0]]
                            # for H_i in HumanNames:
                            #     if HumanNames[best_class_indices[0]] == H_i:
                            result_names = class_names[best_class_indices[0]] if best_class_probabilities[0] > 0.45 else "Unknown"
                            #print(result_names)
                            cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
                            cv2.putText(frame, age_str, (text_x, text_y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 0, 0), thickness=1, lineType=2)
                            cv2.putText(frame, gender_str, (text_x, text_y + 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 0, 0), thickness=1, lineType=2)
                            cv2.putText(frame, race_str, (text_x, text_y + 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 0, 0), thickness=1, lineType=2)

                    else:
                        print('Unable to align')

                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                strs = 'FPS: %2.3f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, strs, (text_fps_x, text_fps_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                # c+=1
                # cv2.putText(frame, "dkasjlk", (text_fps_x, text_fps_y),
                #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            # #video writer
            # out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    _main()
