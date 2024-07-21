from inference_codeformer_function import FaceRestoreHelperWrapper
import cv2

if __name__ == '__main__':
    face_helper_wrapper = FaceRestoreHelperWrapper()
    # basic usage: python3 inference_codeformer.py -w 0.5 --has_aligned -i ./inputs/cropped_faces/
    restored_faces = face_helper_wrapper.call(
        fidelity_weight=0.5, 
        has_aligned=True, 
        # input_path='inputs/cropped_faces/'
        input_path='inputs/cropped_faces/0143.png'
        )
    for f in restored_faces:
        cv2.imshow("result viewer", f)
        cv2.waitKey(100)
# inputs/cropped_faces/0143.png