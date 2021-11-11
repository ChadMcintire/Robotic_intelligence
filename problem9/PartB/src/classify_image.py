# based on https://github.com/experiencor/keras-yolo3
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from yolo3_one_file_to_detect_them_all import *
import tensorflow.keras.preprocessing.image as tfi
from numpy import expand_dims
from keras.models import load_model

def main():
    # save 2nd COCO model from weights
    # save_model("../yolov3.weights", "../yolo_model_2.h5")

    model_loc = '../yolo_model.h5'
    # classify each image
    classify(model_loc, '../input_data/1.jpg')
    classify(model_loc, '../input_data/2.jpg')
    classify(model_loc, '../input_data/3.jpg')
    classify(model_loc, '../input_data/4.jpg')
    classify(model_loc, '../input_data/5.jpg')


# create the model and load weights. Save model
def save_model(weights='../yolov3.weights', model_name='../yolo_model.h5'):
    # define model
    model = make_yolov3_model()
    # load modedef save_model():l weights
    weight_reader = WeightReader(weights)
    # set model weights into model
    weight_reader.load_weights(model)
    # save model to file
    model.save(model_name)

# load the model and classify an image
def classify(model_name, image_file):
    # load yolov3 model
    model = load_model(model_name)
    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    photo_filename = image_file
    # load and prepare image
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
    # make prediction
    yhat = model.predict(image)
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.001
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # define the labels
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    print("v_boxes length: %s" % len(v_boxes))
    if len(v_boxes) == 0:
        print("No objects within threshold of %s" % class_threshold)
    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])
    # draw what we found
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)

# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = tfi.load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = tfi.load_img(filename, target_size=shape)
    # convert to numpy array
    image = tfi.img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='red')
    # show the plot
    pyplot.show()


main()
