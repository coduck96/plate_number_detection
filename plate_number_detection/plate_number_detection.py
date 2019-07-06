import cv2
import numpy as np

net = cv2.dnn.readNetFromTensorflow('./mnist/MINIST_CNN_frozen_graph2.pb')

mode   = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
font = cv2.FONT_HERSHEY_SIMPLEX
x_img = np.zeros(shape=(28, 28), dtype=np.uint8)
x_img=cv2.imread('./1.png')
cv2.imshow('img',x_img)
for i, cnt in enumerate(contours):
#3-1
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(dst, (x, y), (x+width, y+height), (0,0,255), 2)
            cx, cy = x + width/2, y + height/2
            if width > height:
                r = width/2
            else:
                r = height/2            
            cx, cy, r= int(cx), int(cy), int(r)
            img = gray[cy-r:cy+r, cx-r:cx+r]
            img = cv2.resize(img, dsize=(20, 20),interpolation=cv2.INTER_AREA)            
            x_img[:,:] = 0
            x_img[4:24, 4:24] = img
            x_img = cv2.dilate(x_img, None, 2)
            x_img = cv2.erode(x_img, None, 4)
            cv2.imshow('x_img', x_img)
#3-2
            blob = cv2.dnn.blobFromImage(x_img) # blob.shape=(1, 1, 28, 28)
            print('blob.shape=', blob.shape)

            net.setInput(blob)
            res = net.forward()
            print('res=', res)
            y_predict = np.argmax(res, axis = 1)
            print('y_predict=', y_predict)
            digit = int(y_predict[0])
            cv2.putText(dst, str(digit), (x, y), font, 3, (255,0,0), 5)
        
cv2.imshow('dst',dst)
cv2.destroyAllWindows()
