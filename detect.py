from face_detection import RetinaFace


class FaceDetector():
    def __init__(self) -> None:        
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
