from operator import le
from turtle import left
import cv2
import mediapipe as mp

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

        if p1>= 0.95 and p1 <= 1.05 and p2 <= 1.3 and p2 > 1:
            return True
        return False


    
    # def face_orientation(self):
    #     mp_face_mesh = mp.solutions.face_mesh
    #     with mp_face_mesh.FaceMesh(
    #             static_image_mode=True,
    #             max_num_faces=1,
    #             refine_landmarks=True,
    #             min_detection_confidence=0.9) as face_mesh:
    #         results = face_mesh.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
    #         print(results.multi_face_landmarks[0])



if __name__ == "__main__":
    image = cv2.imread("6.jpeg")
    face_extract = face_extractor(image)
    print("\n\n\n\n")
    face_extract.detect_faces()
    if face_extract.number_of_faces == 0:
        print("no face was detected in the image test1.jpeg")
        exit(0)
    elif face_extract.number_of_faces != 1:
        print("too many faces")
        exit(0)
    if not face_extract.face_orientation():
        print("face should be facing the camera")
        exit(0)

    
