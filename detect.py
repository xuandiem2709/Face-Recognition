## Face detection

# Environment:
# - git+https://github.com/elliottzheng/face-detection.git@master

from face_detection import RetinaFace
# from mtcnn_ort import MTCNN

class FaceDetector():
    def __init__(self, gpu_id=0) -> None:
        # MTCNN
        # self.detector = MTCNN()
        
        #  RetinaFace
        self.detector = RetinaFace(
            # gpu_id=gpu_id
        )

    def dict_to_list(kpts_dict):
        keys = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
        kpts_list = []
        for key in keys:
            kpts_list.append(kpts_dict[key])
        return kpts_list

    def __call__(self, image):
        # MTCNN
        # results = self.detector.detect_faces(image)
        # box, landmarks, det_score = []*3
        # for result in results:
        #     bounding_box = result["box"]
        #     keypoints = result['keypoints']
        #     score = round(result['confidence'], 6)
            
        # return box, landmarks, det_score
        #  RetinaFace
        faces = self.detector(image)
        return faces

if __name__ == "__main__":
    # init model
    f = FaceDetector()
    
    # Read frame
    frame = ""

    # Inference
    faces = f(frame)
    for face in faces:
        box, landmarks, det_score = face
        print(box, landmarks, det_score)
