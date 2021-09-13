import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import cv2
import cv2
from google.colab.patches import cv2_imshow
from PIL import ImageColor
import colorsys
import random
import matplotlib.pyplot as plt
from PIL import Image
from YOLO4 import YOLOV41 # YOLOV4 Backbone
from LoadWeights import  WeightReader   # Load pre-trained weights from Darknet FrameWork
from PreprocessImage import * #load_image_pixels  # Load image
from OutputProcess import *


def plotImages2(multipleImages):
    r = random.sample(multipleImages, 3)
    plt.figure(figsize=(20,20))
    plt.subplot(331)
    plt.imshow(cv2.imread(r[0])); plt.axis('off')
    print(r[0])
    plt.subplot(332)
    plt.imshow(cv2.imread(r[1])); plt.axis('off')
    print(r[1])
    plt.subplot(333)
    plt.imshow(cv2.imread(r[2])); plt.axis('off')
    print(r[2])

# cho ảnh đã qua xử lý
# Get boundingbox 
def show_vertebral_label(link, size_reduce, percentreduce,percentreuduce2):
  labels =['Vertebra','Abnormal','Spine','Sacrum']
  # Bước 1: Đọc ảnh, xử lý
  #from PIL import Image
  basewidth = size_reduce
  #img = Image.open('/content/tommy/PHASE2_21_51/1/1338.jpg')
  img= Image.open(link)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  img.save('somepic.jpg')

  photo_filename = 'somepic.jpg'
  image, image_w, image_h = load_image_pixels2(photo_filename, (input_w, input_h))

  # Bước 2: Cho qua YOLO DNN
  model = YOLOV41() # Tạo
  wr = WeightReader('Vert5class.weights')  # Đọc w
  wr.load_weights(model) # Load vào model
  yhat = model.predict(image)

  # Bước 3: Xử lý đầu ra của DNN YOLO --> Kết quả
  boxes = yolo_boxes(yhat)   
  boxes = correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)  
  rs_boxes = do_nms(boxes, 0.5) 
  class_threshold = 0.8
  colors = generate_colors(labels)
  v_boxes_rs, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors) 

  # Bước 4: Vis kết quả
  #draw_boxes3(photo_filename, v_boxes_rs, v_labels, v_scores, v_colors)
  draw_boxes_label('temp.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
  #draw_boxes_label('somepic.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
  #crop_vert_label('temp.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
def abrev(x):
    for i in range(len(x)):
      x[i] = (x[i][1],x[i][0])
    return x
def selection_sort(x):
    temp = []
    x = abrev(x)
    for i in range(len(x)):
        temp.append(x[i][0])
    for i in range(len(x)-1):
        swap =  temp.index(np.min(temp[i:])) # np.argmin(x[i:])
        (temp[i], temp[swap]) = (temp[swap], temp[i])
        (x[i], x[swap]) = (x[swap], x[i])
    x = abrev(x)
    return x
def labelpoint(x,range3):
    temp,temp2, temp3 = [],[],[]
    for i in range(len(x)):
        temp.append(x[i][1])
        temp3.append(x[i][0])
    for i in temp:
      temp2.append(((max(temp3)+range3,i)))
    return temp2
def draw_boxes_label(filename, v_boxes, v_labels, v_scores, percentreduce, percentreuduce2):
    v_colors=['#F657C6','#9BEC1C','#00B2FF','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    leftpoint, rightpoint, centralpoint = [],[],[]
    #print(v_boxes[1])
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        #print(i2)
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        if i2==1:
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)
        if i2==0:
          #print("Đốt Xương {}".format(i))
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)

        # For bounding box
        # For the text background
        color2 = v_colors[i2]
        color2 = ImageColor.getcolor(color2, "RGB")
        color2=tuple(reversed(color2))

        #img = cv2.rectangle(img, (x1, y1), (x2, y2), color2, 3)
        radius = 30; thickness = -1; 
        # coordinates
        center_coordinates = (int(0.5*x2 + 0.5*x1), int(0.5*y2 + 0.5*y1))
        # left rightpoint
        leftp= (x1, int(0.5*y2 + 0.5*y1))
        rightp = (x2, int(0.5*y2 + 0.5*y1))
        he, wi, _ = img.shape
        if v_labels[i]!= "Spine":
          img = cv2.circle(img, center_coordinates, radius, color2, thickness)
          centralpoint.append(center_coordinates)
          leftpoint.append(leftp); rightpoint.append(rightp)
          img = cv2.line(img, center_coordinates, (int(wi/3),center_coordinates[1]) , color2, thickness=3)
          crop = img[y1:y2, x1:x2]
          if v_labels[i] == "Sacrum":
            crop = cv2.resize(crop, (round(crop.shape[0]/1.5), round(crop.shape[1] /1.5)), interpolation=cv2.INTER_AREA)
          try:
            hcrop, wcrop, _ = crop.shape
            x_offset = int(wi/3) - wcrop
            y_offset = center_coordinates[1] - int(hcrop/2)
            #x_end = x_offset + crop.shape[1]
            #y_end = y_offset + crop.shape[0]

            img[y_offset:y_offset+crop.shape[0], x_offset:x_offset+crop.shape[1]] = crop ###
            #x_offset=y_offset=50
            #l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = crop
          except: pass
        # Finds space required by the text so that we can put a background with that amount of width.
        
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)

        # Prints the text. 
        #img = cv2.rectangle(img, (x1, y1-50), (x1 + w, y1), color2, -1)
        text_color="#000000"#v_colors[i2]
        text_color2 = ImageColor.getcolor(text_color, "RGB")
        text_color2 = complement(*text_color2)
        #img = cv2.putText(img, label, (x1, y1 - 4),
        #                    cv2.FONT_HERSHEY_DUPLEX,1.77, text_color2, 2,cv2.LINE_AA)
        # For printing text
        #img = cv2.putText(img, label, (x1, y1),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    #print(centralpoint[:])
    centralpoint = selection_sort(centralpoint)
    labelindex = labelpoint(centralpoint,350)
    #print(centralpoint)
    #print(labelindex)
    leftpoint = selection_sort(leftpoint)
    rightpoint = selection_sort(rightpoint)
    #array2 = np.array(centralpoint).reshape(-1, 1, 2)
    #array2 = np.argsort(array2)
    #print(array2)
    label2 = ['Sacrum','L5','L4','L3','L2','L1',"T12","T11","T10", "T9", "T8", "T7", "T6","T5"]
    for i3 in range(len(centralpoint)-1):
      i4= i3; i3 = len(centralpoint)-1-i3
      img = cv2.line(img, centralpoint[i3], centralpoint[i3-1], (0,255,255), thickness=5)
      img = cv2.line(img, leftpoint[i3], leftpoint[i3-1], (255, 204, 0), thickness=5)
      img = cv2.line(img, rightpoint[i3], rightpoint[i3-1], (255, 204, 0), thickness=5)
      img = cv2.line(img, centralpoint[i3], labelindex[i3], (215, 109, 219), thickness=3)
      img = cv2.line(img, centralpoint[0], labelindex[0], (215, 109, 219), thickness=3)

      (w, h), _ = cv2.getTextSize(label2[i3], cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
      img = cv2.rectangle(img, (labelindex[i4][0]-10, labelindex[i4][1]-h-10), (labelindex[i4][0] + w+10, labelindex[i4][1]+10), (0,255,255), -1)
      (w1, h1), _ = cv2.getTextSize(label2[0], cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
      w1c= labelindex[len(centralpoint)-1][0] ; h1c = labelindex[len(centralpoint)-1][1]
      img = cv2.rectangle(img, (w1c-10, h1c-h1-10), (w1c + w1+10, h1c+10), (0,255,255), -1)
      try:
        #img = cv2.putText(img, label, (x1, y1),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        img = cv2.putText(img, label2[i3], labelindex[i4],
                           cv2.FONT_HERSHEY_DUPLEX,1.77,(0,0,0), 3,cv2.LINE_AA)
        img = cv2.putText(img, label2[0], labelindex[len(centralpoint)-1],
                          cv2.FONT_HERSHEY_DUPLEX,1.77,(0,0,0), 3,cv2.LINE_AA)
      except: pass

      

    #img = cv2.line(img, centralpoint[], 0, (255,255,255), 2)
    #img =cv2.polylines(img,[array2],True,(0,255,255),3)
    cv2.imwrite("result.jpg",img)
    #cv2_imshow(img)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cv2.imread("result.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Test code
#show_vertebral_label('/content/tommy/Phase2/1/1338.jpg',2200,.96,.98)

# Get boundingbox 
def show_vertebral_label2(link, size_reduce, percentreduce,percentreuduce2):
  labels =['Vertebra','Abnormal','Spine','Sacrum']
  # Bước 1: Đọc ảnh, xử lý
  #from PIL import Image
  basewidth = size_reduce
  #img = Image.open('/content/tommy/PHASE2_21_51/1/1338.jpg')
  img= Image.open(link)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  img.save('somepic.jpg')

  photo_filename = 'somepic.jpg'
  image, image_w, image_h = load_image_pixels2(photo_filename, (input_w, input_h))

  # Bước 2: Cho qua YOLO DNN
  model = YOLOV41() # Tạo
  wr = WeightReader('Vert5class.weights')  # Đọc w
  wr.load_weights(model) # Load vào model
  yhat = model.predict(image)

  # Bước 3: Xử lý đầu ra của DNN YOLO --> Kết quả
  boxes = yolo_boxes(yhat)   
  boxes = correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)  
  rs_boxes = do_nms(boxes, 0.5) 
  class_threshold = 0.8
  colors = generate_colors(labels)
  v_boxes_rs, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors) 

  # Bước 4: Vis kết quả
  #draw_boxes3(photo_filename, v_boxes_rs, v_labels, v_scores, v_colors)
  draw_boxes_label2('temp.jpg','somepic.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
  #draw_boxes_label('somepic.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
  #crop_vert_label('temp.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
def abrev(x):
    for i in range(len(x)):
      x[i] = (x[i][1],x[i][0])
    return x
def selection_sort(x):
    temp = []
    x = abrev(x)
    for i in range(len(x)):
        temp.append(x[i][0])
    for i in range(len(x)-1):
        swap =  temp.index(np.min(temp[i:])) # np.argmin(x[i:])
        (temp[i], temp[swap]) = (temp[swap], temp[i])
        (x[i], x[swap]) = (x[swap], x[i])
    x = abrev(x)
    return x
def labelpoint(x,range3):
    temp,temp2, temp3 = [],[],[]
    for i in range(len(x)):
        temp.append(x[i][1])
        temp3.append(x[i][0])
    for i in temp:
      temp2.append(((max(temp3)+range3,i)))
    return temp2

# cho ảnh gốc    
def draw_boxes_label2(filename, filename2, v_boxes, v_labels, v_scores, percentreduce, percentreuduce2):
    v_colors=['#F657C6','#9BEC1C','#00B2FF','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    img2 = cv2.imread(filename2)
    leftpoint, rightpoint, centralpoint = [],[],[]
    #print(v_boxes[1])
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        #print(i2)
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        if i2==1:
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)
        if i2==0:
          #print("Đốt Xương {}".format(i))
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)

        # For bounding box
        # For the text background
        color2 = v_colors[i2]
        color2 = ImageColor.getcolor(color2, "RGB")
        color2=tuple(reversed(color2))

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color2, 3)
        radius = 30; thickness = -1; 
        # coordinates
        center_coordinates = (int(0.5*x2 + 0.5*x1), int(0.5*y2 + 0.5*y1))
        # left rightpoint
        leftp= (x1, int(0.5*y2 + 0.5*y1))
        rightp = (x2, int(0.5*y2 + 0.5*y1))
        he, wi, _ = img.shape
        if v_labels[i]!= "Spine":
          img = cv2.circle(img, center_coordinates, radius, color2, thickness)
          img2 = cv2.circle(img2, center_coordinates, radius, color2, thickness)
          centralpoint.append(center_coordinates)
          leftpoint.append(leftp); rightpoint.append(rightp)
          img2 = cv2.line(img2, center_coordinates, (int(wi/5),center_coordinates[1]) , color2, thickness=3)

          crop = img[y1:y2, x1:x2]
          if v_labels[i] == "Sacrum":
            crop = cv2.resize(crop, (round(crop.shape[0]/1.5), round(crop.shape[1] /1.5)), interpolation=cv2.INTER_AREA)
          try:
            hcrop, wcrop, _ = crop.shape
            x_offset = int(wi/5) - wcrop
            y_offset = center_coordinates[1] - int(hcrop/2)
            #x_end = x_offset + crop.shape[1]
            #y_end = y_offset + crop.shape[0]

            img2[y_offset:y_offset+crop.shape[0], x_offset:x_offset+crop.shape[1]] = crop ###
            #x_offset=y_offset=50
            #l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = crop
          except: pass
        # Finds space required by the text so that we can put a background with that amount of width.
        
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)
        if v_labels[i]!= "Spine":
 
        # Prints the text. 
          img2 = cv2.rectangle(img2, (int(wi/5), center_coordinates[1]-50), (int(wi/5) + w, center_coordinates[1]), color2, -1)
          text_color="#000000"#v_colors[i2]
          text_color2 = ImageColor.getcolor(text_color, "RGB")
          text_color2 = complement(*text_color2)
          img2 = cv2.putText(img2, label, (int(wi/5), center_coordinates[1] - 4),  
                              cv2.FONT_HERSHEY_DUPLEX,1.77, text_color2, 2,cv2.LINE_AA)
          # Object detection # Prints the text. 
          img = cv2.rectangle(img, (x1, y1-50), (x1 + w, y1), color2, -1)
          img = cv2.putText(img, label, (x1, y1 - 4),
                              cv2.FONT_HERSHEY_DUPLEX,1.77, text_color2, 2,cv2.LINE_AA)

          # For printing text
          img2 = cv2.putText(img2, label, (int(wi/5), center_coordinates[1]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    #print(centralpoint[:])
    centralpoint = selection_sort(centralpoint)
    labelindex = labelpoint(centralpoint,350)
    #print(centralpoint)
    #print(labelindex)
    leftpoint = selection_sort(leftpoint)
    rightpoint = selection_sort(rightpoint)
    #array2 = np.array(centralpoint).reshape(-1, 1, 2)
    #array2 = np.argsort(array2)
    #print(array2)
    label2 = ['Sacrum','L5','L4','L3','L2','L1',"T12","T11","T10", "T9", "T8", "T7", "T6","T5"]
    for i3 in range(len(centralpoint)-1):
      i4= i3; i3 = len(centralpoint)-1-i3
      img2 = cv2.line(img2, centralpoint[i3], centralpoint[i3-1], (0,255,255), thickness=5)
      img2 = cv2.line(img2, leftpoint[i3], leftpoint[i3-1], (255, 204, 0), thickness=5)
      img2 = cv2.line(img2, rightpoint[i3], rightpoint[i3-1], (255, 204, 0), thickness=5)
      img2 = cv2.line(img2, centralpoint[i3], labelindex[i3], (215, 109, 219), thickness=3)
      img2 = cv2.line(img2, centralpoint[0], labelindex[0], (215, 109, 219), thickness=3)

      (w, h), _ = cv2.getTextSize(label2[i3], cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
      img2 = cv2.rectangle(img2, (labelindex[i4][0]-10, labelindex[i4][1]-h-10), (labelindex[i4][0] + w+10, labelindex[i4][1]+10), (0,255,255), -1)
      (w1, h1), _ = cv2.getTextSize(label2[0], cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
      w1c= labelindex[len(centralpoint)-1][0] ; h1c = labelindex[len(centralpoint)-1][1]
      img2 = cv2.rectangle(img2, (w1c-10, h1c-h1-10), (w1c + w1+10, h1c+10), (0,255,255), -1)
      try:
        #img = cv2.putText(img, label, (x1, y1),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        img2 = cv2.putText(img2, label2[i3], labelindex[i4],
                           cv2.FONT_HERSHEY_DUPLEX,1.77,(0,0,0), 3,cv2.LINE_AA)
        img2 = cv2.putText(img2, label2[0], labelindex[len(centralpoint)-1],
                          cv2.FONT_HERSHEY_DUPLEX,1.77,(0,0,0), 3,cv2.LINE_AA)
      except: pass

    #img = cv2.line(img, centralpoint[], 0, (255,255,255), 2)
    #img =cv2.polylines(img,[array2],True,(0,255,255),3)
    cv2.imwrite("result.jpg",img2)
    cv2.imwrite("preprocess.jpg",img)
    #cv2_imshow(img)
    #fig = plt.figure(figsize=(10, 10))

    fig, axes = plt.subplots(nrows=1, ncols=3,sharex=True, sharey=True, figsize=(20, 25))
    ax = axes.ravel()
    
    ax[0].imshow(readrgb('somepic.jpg'))
    ax[0].set_title('Original')
    ax[1].imshow(readrgb('preprocess.jpg'))
    ax[1].set_title('Object Detection YOLO4')
    ax[2].imshow(readrgb('result.jpg'))
    ax[2].set_title('Labelling & Dignoses')

    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()

    #plt.imshow(cv2.cvtColor(cv2.imread("result.jpg"), cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()


def readrgb(link):
      pic = cv2.imread(link)
      pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
      return pic



# Get boundingbox 
def show_vertebral_label3(link, size_reduce, percentreduce,percentreuduce2):
  labels =['Vertebra','Abnormal','Spine','Sacrum']
  # Bước 1: Đọc ảnh, xử lý
  #from PIL import Image
  basewidth = size_reduce
  #img = Image.open('/content/tommy/PHASE2_21_51/1/1338.jpg')
  img= Image.open(link)
  wpercent = (basewidth/float(img.size[0]))
  hsize = int((float(img.size[1])*float(wpercent)))
  img = img.resize((basewidth,hsize), Image.ANTIALIAS)
  img.save('somepic.jpg')

  photo_filename = 'somepic.jpg'
  image, image_w, image_h = load_image_pixels2(photo_filename, (input_w, input_h))

  # Bước 2: Cho qua YOLO DNN
  model = YOLOV41() # Tạo
  wr = WeightReader('Vert5class.weights')  # Đọc w
  wr.load_weights(model) # Load vào model
  yhat = model.predict(image)

  # Bước 3: Xử lý đầu ra của DNN YOLO --> Kết quả
  boxes = yolo_boxes(yhat)   
  boxes = correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)  
  rs_boxes = do_nms(boxes, 0.5) 
  class_threshold = 0.8
  colors = generate_colors(labels)
  v_boxes_rs, v_labels, v_scores, v_colors = get_boxes(boxes, labels, class_threshold, colors) 

  # Bước 4: Vis kết quả
  #draw_boxes3(photo_filename, v_boxes_rs, v_labels, v_scores, v_colors)
  draw_boxes_label3('temp.jpg','somepic.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
  #draw_boxes_label('somepic.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
  #crop_vert_label('temp.jpg', v_boxes_rs, v_labels, v_scores, percentreduce, percentreuduce2)
def abrev(x):
    for i in range(len(x)):
      x[i] = (x[i][1],x[i][0])
    return x

def draw_boxes_label3(filename, filename2, v_boxes, v_labels, v_scores, percentreduce, percentreuduce2):
    v_colors=['#F657C6','#9BEC1C','#00B2FF','#FADD3A','#A2E24D','#CA0F3B','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD','#DE1F55',"#F0326A","#CAFD65", '#3CC983','#4600CD']
    img = cv2.imread(filename)
    img2 = cv2.imread(filename2)
    leftpoint, rightpoint, centralpoint = [],[],[]
    #print(v_boxes[1])
    for i in range(len(v_boxes)):
        labels =['Vertebra','Abnormal','Spine','Sacrum']
        i2 = labels.index(v_labels[i])
        #print(i2)
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        label = "%s:%.0f" % (v_labels[i], v_scores[i]) + "%"
        if i2==1:
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)
        if i2==0:
          #print("Đốt Xương {}".format(i))
          y1,y2 = int(percentreduce*y1), int(percentreuduce2*y2)

        # For bounding box
        # For the text background
        color2 = v_colors[i2]
        color2 = ImageColor.getcolor(color2, "RGB")
        color2=tuple(reversed(color2))

        #img = cv2.rectangle(img, (x1, y1), (x2, y2), color2, 3)
        radius = 30; thickness = -1; 
        # coordinates
        center_coordinates = (int(0.5*x2 + 0.5*x1), int(0.5*y2 + 0.5*y1))
        # left rightpoint
        leftp= (x1, int(0.5*y2 + 0.5*y1))
        rightp = (x2, int(0.5*y2 + 0.5*y1))
        he, wi, _ = img.shape
        if v_labels[i]!= "Spine":
          img = cv2.circle(img, center_coordinates, radius, color2, thickness)
          img2 = cv2.circle(img2, center_coordinates, radius, color2, thickness)
          centralpoint.append(center_coordinates)
          leftpoint.append(leftp); rightpoint.append(rightp)
          img2 = cv2.line(img2, center_coordinates, (int(wi/5),center_coordinates[1]) , color2, thickness=3)

          crop = img[y1:y2, x1:x2]
          if v_labels[i] == "Sacrum":
            crop = cv2.resize(crop, (round(crop.shape[0]/1.5), round(crop.shape[1] /1.5)), interpolation=cv2.INTER_AREA)
          try:
            hcrop, wcrop, _ = crop.shape
            x_offset = int(wi/5) - wcrop
            y_offset = center_coordinates[1] - int(hcrop/2)
            #x_end = x_offset + crop.shape[1]
            #y_end = y_offset + crop.shape[0]

            img2[y_offset:y_offset+crop.shape[0], x_offset:x_offset+crop.shape[1]] = crop ###
            #x_offset=y_offset=50
            #l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = crop
          except: pass
        # Finds space required by the text so that we can put a background with that amount of width.
        
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 2)
        if v_labels[i]!= "Spine":
        # Prints the text. 
          img2 = cv2.rectangle(img2, (int(wi/5), center_coordinates[1]-50), (int(wi/5) + w, center_coordinates[1]), color2, -1)
          text_color="#000000"#v_colors[i2]
          text_color2 = ImageColor.getcolor(text_color, "RGB")
          text_color2 = complement(*text_color2)
          img2 = cv2.putText(img2, label, (int(wi/5), center_coordinates[1] - 4),  
                              cv2.FONT_HERSHEY_DUPLEX,1.77, text_color2, 2,cv2.LINE_AA)
          # For printing text
          img2 = cv2.putText(img2, label, (int(wi/5), center_coordinates[1]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    #print(centralpoint[:])
    centralpoint = selection_sort(centralpoint)
    labelindex = labelpoint(centralpoint,350)
    #print(centralpoint)
    #print(labelindex)
    leftpoint = selection_sort(leftpoint)
    rightpoint = selection_sort(rightpoint)
    #array2 = np.array(centralpoint).reshape(-1, 1, 2)
    #array2 = np.argsort(array2)
    #print(array2)
    label2 = ['Sacrum','L5','L4','L3','L2','L1',"T12","T11","T10", "T9", "T8", "T7", "T6","T5"]
    for i3 in range(len(centralpoint)-1):
      i4= i3; i3 = len(centralpoint)-1-i3
      img2 = cv2.line(img2, centralpoint[i3], centralpoint[i3-1], (0,255,255), thickness=5)
      img2 = cv2.line(img2, leftpoint[i3], leftpoint[i3-1], (255, 204, 0), thickness=5)
      img2 = cv2.line(img2, rightpoint[i3], rightpoint[i3-1], (255, 204, 0), thickness=5)
      img2 = cv2.line(img2, centralpoint[i3], labelindex[i3], (215, 109, 219), thickness=3)
      img2 = cv2.line(img2, centralpoint[0], labelindex[0], (215, 109, 219), thickness=3)

      (w, h), _ = cv2.getTextSize(label2[i3], cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
      img2 = cv2.rectangle(img2, (labelindex[i4][0]-10, labelindex[i4][1]-h-10), (labelindex[i4][0] + w+10, labelindex[i4][1]+10), (0,255,255), -1)
      (w1, h1), _ = cv2.getTextSize(label2[0], cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
      w1c= labelindex[len(centralpoint)-1][0] ; h1c = labelindex[len(centralpoint)-1][1]
      img2 = cv2.rectangle(img2, (w1c-10, h1c-h1-10), (w1c + w1+10, h1c+10), (0,255,255), -1)
      try:
        #img = cv2.putText(img, label, (x1, y1),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        img2 = cv2.putText(img2, label2[i3], labelindex[i4],
                           cv2.FONT_HERSHEY_DUPLEX,1.77,(0,0,0), 3,cv2.LINE_AA)
        img2 = cv2.putText(img2, label2[0], labelindex[len(centralpoint)-1],
                          cv2.FONT_HERSHEY_DUPLEX,1.77,(0,0,0), 3,cv2.LINE_AA)
      except: pass

      

    #img = cv2.line(img, centralpoint[], 0, (255,255,255), 2)
    #img =cv2.polylines(img,[array2],True,(0,255,255),3)
    cv2.imwrite("result.jpg",img2)
    #cv2_imshow(img)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(cv2.imread("result.jpg"), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


