# Real-time Facial Recognition using YOLO and FaceNet

## Available Funtions
* **Face Alignment:** - YOLO v3.
* **Training on FaceNet:** Transfer learning
* **Real-time Facial Recognition:** OpenCV rendering

## Configuration
* OS: Windows 10 / Ubuntu 18.04
* GPU: NVIDIA GeForce GTX 1060
* CUDA TOOLKIT: v9.0
* cuDNN SDK: v7.5 (corresponding to CUDA TOOLKIT v9.0)
* Python: 3.x
* tensorflow-gpu: 1.10.1


1. **Face Alignment.**

     You can use ```align_dataset_yolo_gpu.py```.
     
     ***NEED TO DOWNLOAD MANUALLY***
     First, use ```get_models.sh``` in \align\yolo_weights\ to get the pre-trained model of YOLO.
     
     Then create a folder in \align and name it as "unaligned_faces", put all your images in this folder. In \align\unaligned_faces, one person has one folder with his/her name as folder name and all his/her images should be put in the corresponding folder. 
     
     Finally run
     ```bash
     $ python align_dataset_yolo_gpu.py
     ```
     
     The result will be generated in \aligned_faces folder, copy all the results with the class folder to /output folder for later use.
     
2. **Training FaceNet Model**    ***Currently Skipped***

     * If you want to directly use a pre-trained model for facial recognition, just skip this step.
     * If you want to implement a tranfer learning with a pre-trained model and your own dataset, you need to first download this pre-trained [model](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit), put it in /models and unzip it. Make sure that the directory /models/20170512-110547 has 4 files.
       
       Then run
       ```bash
       $ python train_tripletloss.py
       ```
     
       The trained model will be in the /models/facenet.
     
     * If you want to train your own model from scratch. In ```train_tripletloss.py``` line 433, there is an optional argument named "--pretrained_model", delete its default value.
     
       Then run again 
       ```bash
       $ python train_tripletloss.py
       ```
     
       The trained model will also be in the /models/facenet.

3. **Training SVM Model**

     In ```Make_classifier_knnForFaces.py``` / ```Make_classifier_svmForFaces.py```,  change the "modeldir" variable to your own FaceNet model path. If you have trained a model already, just use the corresponding path, otherwise there are several pre-trained model you can use:
     
     VGGFace2: [20180402-114759](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view)
     
     Then run
     ```bash
       $ python Make_classifier_svmForFaces.py
     ```
     
     The SVM model will be generated in \myclassifier.
     
4. **Real-time Facial Recognition**

     There are two versions —  ```realtime_facenet_yolo_gpu_2.py``` and ```realtime_facenet_yolo_gpu_http_streaming2.py``` .
     
     First Modify the "url_of_ip_camera" variable in ```realtime_facenet_yolo_gpu_http_streaming2.py``` to do real time streamming with an IP camera.
     
     Then run
     ```bash
       $ python realtime_facenet_yolo_gpu_2.py
     ```
     
     or
     
     ```bash
       $ python realtime_facenet_yolo_gpu.py
     ```
     

## References

A special thanks to the following:

* davidsandberg https://github.com/davidsandberg/facenet

  Provided FaceNet code for training and embedding generation


* sthanhng https://github.com/sthanhng/yoloface

  Provided a YOLO model trained on WIDER FACE for real-time facial detection


* cryer https://github.com/cryer/face_recognition

  Provided a framework for moving images from webcam to model, model to real-time on-screen bounding boxes and names


* https://github.com/Tony607/Keras_age_gender?fbclid=IwAR3g_k362ygrSRIKMttW0vzFP_G00juSUmUQywuu5BHsZ9p41u5JuDkpAbI

  Provided a framework for easy real time gender age prediction from webcam video with Keras