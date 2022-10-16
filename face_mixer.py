from ast import arg
import cv2
import mediapipe as mp
import numpy as np
import argparse


class face_extractor():
    def __init__(self,image):
        self.image = image
        self.number_of_faces = None
        self.faces_simple_landmarks = None
        self.hight, self.width, _= image.shape
    def detect_faces(self):
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.9) as face_detection:
            results = face_detection.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            self.number_of_faces = len(results.detections)
            self.faces_simple_landmarks = results.detections
    def face_orientation(self):
        face = self.faces_simple_landmarks[0]
        mp_face_detection = mp.solutions.face_detection
        nose = mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.NOSE_TIP)
        right_eye = mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        left_eye = mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.LEFT_EYE)
        mouth = mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.MOUTH_CENTER)
        
        nose_cord = {
            "x":nose.x*self.width,
            "y":nose.y*self.hight
        }
        r_eye_cord = {
            "x":right_eye.x*self.width,
            "y":right_eye.y*self.hight
        }
        l_eye_cord = {
            "x":left_eye.x*self.width,
            "y":left_eye.y*self.hight
        }
        mouth_cord = {
            "x":mouth.x*self.width,
            "y":mouth.y*self.hight
        }

        a1 = (l_eye_cord["y"] - r_eye_cord["y"])/(l_eye_cord["x"]-r_eye_cord["x"])
        b1 = l_eye_cord["y"] - a1 * l_eye_cord["x"]
        a2 = (mouth_cord["y"] - nose_cord["y"])/(mouth_cord["x"] - nose_cord["x"])
        b2 = mouth_cord["y"] - a2*mouth_cord["x"]

        third_eye = {}
        third_eye["x"] = (b2 - b1)/(a1-a2)
        third_eye["y"] = a1*third_eye["x"]+b1


        r_eye_dis = ((nose_cord["x"] - r_eye_cord["x"])**2 + (nose_cord["y"] - r_eye_cord["y"])**2)**0.5
        l_eye_dis = ((nose_cord["x"] - l_eye_cord["x"])**2 + (nose_cord["y"] - l_eye_cord["y"])**2)**0.5
        mouth_dis = ((nose_cord["x"] - mouth_cord["x"])**2 + (nose_cord["y"] - mouth_cord["y"])**2)**0.5
        third_eye_dis = ((nose_cord["x"] - third_eye["x"])**2 + (nose_cord["y"] - third_eye["y"])**2)**0.5

        p1 = round(r_eye_dis/l_eye_dis,2)
        p2 = round(third_eye_dis/mouth_dis,1)
        if p1>= 0.8 and p1 <= 1.2 and p2 <= 1.6 and p2 >= 0.9:
            return True
        return False

class face_mixer():
    def __init__(self,image1,image2):
        self.image1 = image1
        self.backup1 = image1.copy()
        self.image2 = image2
        self.backup2 = image2.copy()
        self.output = None
    def background_blure(self):
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
            results = selfie_segmentation.process(cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB))
            mask  = (np.stack((results.segmentation_mask,)* 3,axis=-1)*255).astype(np.uint8)
            mask = cv2.GaussianBlur(mask,(25,25),200)/255
            blured_img = cv2.GaussianBlur(self.image1,(25,25),200)
            rev_mask = 1-mask
            self.image1 = (self.image1 * mask + blured_img * rev_mask).astype(np.uint8)
    def extract_face_and_merge(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8) as face_mesh:

            results = face_mesh.process(cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB))
            annotated_image = self.image2 * 0
            face_landmarks= results.multi_face_landmarks[0]
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            _, im_th = cv2.threshold(annotated_image,200,255,cv2.THRESH_BINARY)
            h,w,_ = im_th.shape
            mk = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_th,mk,(0,0),(255,255,255))
            im_th = (255 - im_th)
            mask = cv2.GaussianBlur(im_th,(55,55),200) / 255
            # mask = im_th / 255
            self.image2 = (mask * self.image2).astype(np.uint8)


            results0 = face_mesh.process(cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB))
            annotated_image0 = self.image1 * 0
            face_landmarks0= results0.multi_face_landmarks[0]
            mp_drawing.draw_landmarks(
                image=annotated_image0,
                landmark_list=face_landmarks0,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            _, im_th0 = cv2.threshold(annotated_image0,200,255,cv2.THRESH_BINARY)
            h0,w0,_ = im_th0.shape
            mk0 =  np.zeros((h0+2, w0+2), np.uint8)
            cv2.floodFill(im_th0,mk0,(0,0),(255,255,255))
            mask0 = cv2.GaussianBlur(im_th0,(55,55),200) / 255
            # mask0 = im_th0 / 255
            self.image1 = (mask0 * self.image1).astype(np.uint8)

            lm = np.array([[l.x * w, l.y * h] for l in face_landmarks.landmark],np.float32).reshape(-1,1,2)
            lm0 = np.array([[l.x * w0, l.y * h0] for l in face_landmarks0.landmark],np.float32).reshape(-1,1,2)

            M, m = cv2.findHomography(lm, lm0, 0,confidence = 1)
            im_dst = cv2.warpPerspective(self.image2, M,(w0,h0))
            self.backup2 = cv2.warpPerspective(self.backup2, M,(w0,h0))
            overlaping_mask = (im_dst.astype(np.uint64) + self.image1.astype(np.uint64) )>255
            overlaping_mask = np.array([[[j.any()]*3 for j in i] for i in overlaping_mask])
            im_dst = im_dst + self.image1
            self.output = np.where(overlaping_mask, self.backup2, im_dst)


  



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", help="the main face image",required=True)
    parser.add_argument("-s", help="the secondary face image",required=True)
    parser.add_argument("-o", help="output image",default="output.jpeg")
    args = parser.parse_args()

    image1 = cv2.imread(args.m)
    image2 = cv2.imread(args.s)
    face_extract1 = face_extractor(image1)
    face_extract2 = face_extractor(image2)
    face_extract1.detect_faces()
    if face_extract1.number_of_faces == 0:
        print("no face was detected in the image1")
        exit(0)
    elif face_extract1.number_of_faces != 1:
        print("too many faces in image1")
        exit(0)
    if not face_extract1.face_orientation():
        print("face should be facing the camera image1")
        exit(0)

    face_extract2.detect_faces()
    if face_extract2.number_of_faces == 0:
        print("no face was detected in the image2")
        exit(0)
    elif face_extract2.number_of_faces != 1:
        print("too many faces in image2")
        exit(0)
    if not face_extract2.face_orientation():
        print("face should be facing the camera image2")
        exit(0)
    


    face_m = face_mixer(image1,image2)
    face_m.background_blure()
    face_m.extract_face_and_merge()
    cv2.imwrite(args.o, face_m.output)
    cv2.imshow("output",face_m.output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
