from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.cluster import KMeans
import numpy as np
import cv2


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu')) 
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

############################################################
def myFunc(img):
    p=1;
    kernel = np.ones((7,7),np.float32)/25
    img1 = cv2.filter2D(img,-1,kernel)
    #cv2.imshow('Gaussian Image',img1)
    
    ## Bilateral Filter for Edge Enhancement
    img3 = cv2.bilateralFilter(img1,9,75,75)
    #cv2.imshow('Bilateral Filtered Image',img3)
    
    
    ## RGB to Gray conversion
    GRAY_Img = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('GRAY Image',GRAY_Img)
    
    Data2Ext=GRAY_Img;
    #cv2.imwrite('ImageRedist.jpg',Data2Ext);
    
    
    roi1=GRAY_Img;
    r,c=roi1.shape;
    if p==1:
        roi = roi1.reshape((roi1.shape[0] * roi1.shape[1], 1))
    
    ## KMEANS clustering
    imgkmeans = KMeans(n_clusters=3, random_state=0);
    imgkmeans.fit(roi);
    label_values=imgkmeans.labels_;
    Label_reshped = np.reshape(label_values,(roi1.shape[0] ,roi1.shape[1]));
    
    segmentregions=roi1;
    
    rows,cols = roi1.shape;
    # Thresholding for segmentation
    for i in range(0,rows):
        for j in range(0,cols):
            pixl=Label_reshped[i,j];
            if pixl==0:
                segmentregions[i,j]=255;
                
            else:
                segmentregions[i,j]=0;
    
    
    # Thresholding for segmentation
    NewImage = Data2Ext;
    NewImage= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    extractedregions=NewImage;
    
    for k in range(0,rows):
        for l in range(0,cols):
            pixl1=segmentregions[k,l];
            if pixl1==0:
                extractedregions[k,l]=NewImage[k,l];
            else:
                extractedregions[k,l]=0;
    
    img=cv2.cvtColor(extractedregions,cv2.COLOR_GRAY2RGB)
    return img
#########################################################
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=myFunc)

test_datagen = ImageDataGenerator(rescale=1./255,
                                  preprocessing_function=myFunc)

print('Loading training set:');
training_set = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

print('Loading testing set:');
test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=210, 
        epochs=50,
        validation_data=test_set,
        validation_steps=54) 

classifier.save('model.h5')