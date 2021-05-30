                            ###Object Detection using YoloV3 and OpenCV###
							                    

import cv2
import numpy as np
import time

def load_yolo():
	net = cv2.dnn.readNet("C:\\Users\\hp\\Desktop\\yolo-coco\\yolov3.weights",
	"C:\\Users\\hp\\Desktop\\yolo-coco\\yolov3.cfg")
	classes = []
	with open("C:\\Users\\hp\\Desktop\\yolo-coco\\coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(5, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.9, fy=0.8)
	height, width, channels = img.shape
	return img, height, width, channels

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
		    
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			conf=str(round(confs[i],2))
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
			cv2.putText(img, label+" "+conf, (x, y - 5), font, 1, color, 2)
	cv2.imshow("Image", img)

def image_detect(img_path): 
	model, classes, colors, output_layers = load_yolo()
	image, height, width, channels = load_image(img_path)
	blob, outputs = detect_objects(image, model, output_layers)
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	while True:

		key = cv2.waitKey(1)
		if key == 27: #ESC 
			break

def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()

#image_detect('C:\\Users\\hp\\Desktop\\yolo-coco\\kitchen.jpg')
#image_detect('C:\\Users\\hp\\Desktop\\yolo-coco\\image5.jpg') 
#image_detect('C:\\Users\\hp\\Desktop\\yolo-coco\\car.jpg')
#image_detect('C:\\Users\\hp\\Desktop\\yolo-coco\\image2.jpg')

#image_detect('C:\\Users\\hp\\Desktop\\yolo-coco\\home.jpg')
image_detect('C:\\Users\\hp\\Desktop\\yolo-coco\\image6.jpg')

#start_video("C:\\Users\\hp\\Desktop\\yolo-coco\\myvideo.mp4")