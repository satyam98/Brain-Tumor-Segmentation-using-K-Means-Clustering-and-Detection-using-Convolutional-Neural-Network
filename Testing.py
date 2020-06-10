from keras.models import load_model
from sklearn.cluster import KMeans
import cv2
import numpy as np

model = load_model('model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('Testing/1_SN8nPp94AYQslNUyDGkmqA.jpeg')
# EXTRACT TUMOR START
########################################################
###################################################
p=1;
kernel = np.ones((7,7),np.float32)/25
img1 = cv2.filter2D(img,-1,kernel)
cv2.imshow('Gaussian Image',img1)

## Bilateral Filter for Edge Enhancement
img3 = cv2.bilateralFilter(img1,9,75,75)
cv2.imshow('Bilateral Filtered Image',img3)


## RGB to Gray conversion
GRAY_Img = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
cv2.imshow('GRAY Image',GRAY_Img)

Data2Ext=GRAY_Img;
cv2.imwrite('ImageRedist.jpg',Data2Ext);

# APPLY THRESHOLDING
ret,th1 = cv2.threshold(img3,127,255,cv2.THRESH_BINARY)
cv2.imshow('after thresholding.jpg',th1);

roi1=GRAY_Img;
r,c=roi1.shape;
if p==1:
    roi = roi1.reshape((roi1.shape[0] * roi1.shape[1], 1))

## KMEANS clustering
imgkmeans = KMeans(n_clusters=3, random_state=1);
imgkmeans.fit(roi);
label_values=imgkmeans.labels_;
Label_reshped = np.reshape(label_values,(roi1.shape[0] ,roi1.shape[1]));

segmentregions=roi1;
blobregions=roi1;

rows,cols = roi1.shape;
# Thresholding for segmentation
for i in range(0,rows):
    for j in range(0,cols):
        pixl=Label_reshped[i,j];
        if pixl==0:
            
            
            
            segmentregions[i,j]=255;
            
        else:
            segmentregions[i,j]=0;

cv2.imshow('Segemented Image',segmentregions)


# Thresholding for segmentation
NewImage = Data2Ext;
NewImage= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
extractedregions=NewImage;

for k in range(0,rows):
    for l in range(0,cols):
        pixl1=segmentregions[k,l];
        if pixl1==0:
##            print 'ok'
            extractedregions[k,l]=NewImage[k,l];
##            print extractedregions[k,l]
        else:
##            print 'no'
            extractedregions[k,l]=0;

cv2.imshow('Extracted Regions Image',extractedregions)

#########################################################
img=cv2.cvtColor(extractedregions,cv2.COLOR_GRAY2RGB)
# EXTRACT TUMOR END

img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])

classes = model.predict_classes(img)

if classes[0][0] == 1:
    prediction = 'NORMAL'
else:
    prediction = 'ABNORMAL'
    
print (prediction)