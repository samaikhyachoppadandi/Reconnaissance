# Reconnaissance
Image Segmentation using Segnet and U-Net on SAR images

Space technologies can give operational advantage over potential adversaries. In predominant areas like defense and science, advance space engineering requires an automatic feature labelling system to accurately classify features in overhead imagery which helps in effective satellite intelligence and surveillance from space. Using computer vision, we can also orchestrate responses to future threats and opportunities with enhanced and affordable defense and security capability.![image](https://user-images.githubusercontent.com/37493247/194464814-7bf9937a-d386-4c6a-9a1b-5e4e4c016d8a.png)

The dataset can be retrieved from https://www.kaggle.com/competitions/dstl-satellite-imagery-feature-detection/data
The dataset consists of 
3- Band Images -  RGB natural color images
16 Band Images - Multispectral (400 – 1040nm) and short-wave infrared (1195-2365nm) range. 
Grid_sizes – Xmin and Y max coordinates of each image
Polygon_wkt – Classes and polygon coordinates

![image](https://user-images.githubusercontent.com/37493247/194465046-18339acd-d93e-43f1-9077-3efdcef88b3f.png)  ![image](https://user-images.githubusercontent.com/37493247/194465077-60d535ab-6e56-4115-8507-19e5f8b53912.png)

The dataset is trained on U-net and Segnet and we observe the following results
The proposed algorithms are trained with the prepared datasets. 
The learning rate of 0.0001 was applied for both the algorithms, with 15 epochs each. 
The outputs generated were considerable as the maximum classes were forest and manmade structures.

One of U-Net outputs:

![image](https://user-images.githubusercontent.com/37493247/194465728-94ee6bd2-46d6-491a-983e-acfacd09ad0b.png)

One of Segnet outputs:

![image](https://user-images.githubusercontent.com/37493247/194465747-106baabe-632a-4812-872d-ef8d29ef4d60.png)
