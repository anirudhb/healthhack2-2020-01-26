# import dlib
import PIL.Image as Image
from skimage import io
import matplotlib.pyplot as plt
import tensorflow as tf 

import train

class Predictor:
    def __init__(self):
        self.base_model = train.get_backbone_model()
        self.model = train.get_classification_model(self.base_model)
        self.model.load_weights("./curr_best_model.h5")
        # self.face_detector = dlib.get_frontal_face_detector()
        self.status = ["All Good", "Model unsure, please check manually"]
    def detect_faces(self, image):
        # Run detector and get bounding boxes of the faces on image.
        detected_faces = self.face_detector(image, 1)
        face_frames = [(x.left(), x.top(),
                        x.right(), x.bottom()) for x in detected_faces]
        
        return face_frames[0]
    def get_emotion_image(self, image):
        detected_face = self.detect_faces(image)
        face = Image.fromarray(image).crop(detected_face)
        idx = tf.math.argmax(self.model.predict(face))
        return (train.total_classes[idx], status)
    
    def get_emotion_file(self, image_file):
        image = tf.expand_dims(io.imread(image_file), 0)
        # detected_face = self.detect_faces(image)
        # face = Image.fromarray(image).crop(detected_face)
        idx = tf.math.argmax(self.model.predict([image])[0])
        return train.total_classes[idx]