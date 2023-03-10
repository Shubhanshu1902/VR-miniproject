# VR Mini-Project Intermediate Report

#### Team Members: 

- Prakhar Rastogi IMT2020052
- Vatsal Dhama IMT2020029
- Shubhanshu Agrawal IMT2020078

#### GitHub Link: 

## Question 3a:

For now, we have created a convolutional neural network with the following architecture:

	- 2 Convolution layer
	- 2 Fully connected layers

```python
Sequential(
  (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (4): ReLU()
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Flatten(start_dim=1, end_dim=-1)
  (7): Linear(in_features=400, out_features=120, bias=True)
  (8): ReLU()
  (9): Linear(in_features=120, out_features=10, bias=True)
)
```



In the future we will explore a bit more, adding more layers and customizing the parameters.

Initially we set the learning rate to 0.1, but we were getting very low accuracies in the range 10%. We played around with the parameters a bit and finally, for now, we have set the learning rate to be 0.01 and the batch size of 256. We ran our model for 20 epochs. The following are the results we obtained on using tanh/ReLU/Sigmoid:

- **Using Sigmoid:**
  - Classification performance (accuracy): 78%
  - ![image-20230310221327253](C:\Users\rprak\AppData\Roaming\Typora\typora-user-images\image-20230310221327253.png)

- **Using Tanh:**
  - Classification performance (accuracy):  64%
  - ![image-20230310222052675](C:\Users\rprak\AppData\Roaming\Typora\typora-user-images\image-20230310222052675.png)

- **Using ReLU:**
  - Classification performance (accuracy): 60%
  - ![image-20230310223344958](C:\Users\rprak\AppData\Roaming\Typora\typora-user-images\image-20230310223344958.png)

## Question 3c:

We went through both of the YOLO V1 and YOLO V2 papers. The following are 5 additional features we found in YOLO V2 that addresses few of the limitations of YOLO V1:

- High-resolution classifier: YOLO V2 implements a high-res classifier that works on 448X448 input image instead of the 224x224 image input used in YOLO V1. This helps in improved detection of small objects.

- Batch Normalization: Contrary to YOLO V1, YOLO V2 uses batch normalization, which helps in stabilizing the training process. Thus, the resulting network will be more stable and accurate.

- Multi-scale training: In YOLO V2, the network is trained on various images of varying sizes. This allows the network to learn how to detect object at different scales. Thus, leading to significant improvement in accuracy.

- Anchor boxes: In YOLO V2, the concept of anchor boxes is introduced. anchors are predefined bounding boxes of varying sizes and shapes. The network then uses these anchors to predict object locations, leading to better localization of objects and increases accuracy.

- Passthrough Layers:  These allow the network to use features from earlier layers to make better predictions.

  

## Question 3d:

For now we have implemented YOLO V2 for vehicle detection. In the future, we will implement Faster RCNN. 

Our working is as follows:

- We used the pretrained YOLO V2 model from the pytorch library.
- The model is trained on the Microsoft's famous [COCO dataset.](https://cocodataset.org/#download).
- For each frame, we find the objects in it. 
- Out of the all of them, we filter out the vehicles with classes: car, truck, motorcycle, bicycle.
- Using the (X_min, Y_min) and (X_max, Y_max)  we detected starting and ending points for the bounding boxes. 
- Each of the bounding box has the class name and confidence score on top of it.
- The confidence score is rounded off to 2 decimal places for better accuracy.

Soon, we'll implement how many cars pass the highway functionality too. We have implemented the total vehicle counter in frame.

A few results:



![image-20230310230409768](C:\Users\rprak\AppData\Roaming\Typora\typora-user-images\image-20230310230409768.png)

![image-20230310230424248](C:\Users\rprak\AppData\Roaming\Typora\typora-user-images\image-20230310230424248.png)

![image-20230310230544352](C:\Users\rprak\AppData\Roaming\Typora\typora-user-images\image-20230310230544352.png)