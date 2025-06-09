# Object-Detection-with-YOLO
Deep Learning course assignment # 02 : Object Detection with YOLO

## Task 1: Implementing YOLO on a Pre-trained Model 

 
Sample 1 input image:

![image](https://github.com/user-attachments/assets/f952e96f-9bfb-4401-bc60-d572c4c1ec15)


Sample 1 output image:

![image](https://github.com/user-attachments/assets/001ce28a-65b3-4b7a-a115-b89866d59740)


Sample 2 input image:

 ![image](https://github.com/user-attachments/assets/2db69a22-aaf5-46c8-a928-d045115186a7)

Sample 2 output image:

 ![image](https://github.com/user-attachments/assets/4a41b683-5646-4a93-a569-633a85fa54db)


Sample output video screenshot: please check the snapshot of sample output video complete video is provided at github link.
 ![image](https://github.com/user-attachments/assets/37d5293b-c673-4e98-8f34-7877c7c81540)

Sample video input and output images and python code: https://github.com/Rafia-Shaikh-eng/Object-Detection-with-YOLO/tree/da142683ee316851d2e592c0904dbabc099c0b6c/Task-01
 
## Task 2: Fine-Tuning YOLO on a Custom Dataset 
### Dataset description:
 
I have downloaded face mask dataset from kaggle. There are 853 images and each image’s annotated file in xml format.
 ![image](https://github.com/user-attachments/assets/c7319b3d-6433-49d1-b1a0-0431e56378c7)

Below figure is the snapshot of YOLOv8 YAML configuration file used to define the dataset structure. There are 03 classes: with mask, without mask and mask weared incorrect.
 ![image](https://github.com/user-attachments/assets/6aa517a7-8846-4a61-b322-7987031429c0)

### Model training:
 ![image](https://github.com/user-attachments/assets/39ee9172-71d5-4739-9966-f5a4d6e6affc)

### Model prediction:


![image](https://github.com/user-attachments/assets/355b7ed9-f0d4-44cf-8ea2-20e61ae3386b)


### Model weights:
![image](https://github.com/user-attachments/assets/bc7d2453-152e-42bc-802b-ad2969ccf512)

 
## Task#02 code: https://www.kaggle.com/code/rafia61/face-mask-detection-yolov8/edit

Results:
  
![image](https://github.com/user-attachments/assets/f0c5d391-3179-45ca-8d5a-8e700517496d)

 


### Validation prediction output images:
 

![image](https://github.com/user-attachments/assets/7e208994-316d-4096-8fb7-04639f68c4ad)

![image](https://github.com/user-attachments/assets/47fab897-88c7-4951-bf01-c04455ba3822)

 
## Task 3: Real-Time Object Detection
Below is real-time face mask detection video’s screenshots.
There were 03 classes the output of each class in show in the attached screenshots. I have use the same model (best.pt also known as earned parameters) which I trained in the task#02 using it in this real-time object task 3.

![image](https://github.com/user-attachments/assets/6630caaf-512e-47e3-87d7-50477eb8cf1a)


Link of demo output video of real time face mask detection: https://drive.google.com/drive/folders/1jcAGuTlnDiVM11UtKBKFjizH1chNSV3B?usp=drive_link

Link of Code + input data+ sample output images/videos:
https://github.com/Rafia-Shaikh-eng/Object-Detection-with-YOLO/tree/main

