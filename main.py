import cv2
import time
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
import os
from FaceRestorationHelperWrapper import FaceRestoreHelperWrapper
import numpy as np

def load_faces(app):
    source_dir = "./faceImages"
    file_list = os.listdir(source_dir)
    face_list = []
    for filename in file_list:
        if filename.endswith('.jpg'):
            img = cv2.imread(f"{source_dir}/{filename}")
            faces = app.get(img)
            assert len(faces) == 1
            face_list.append(faces[0])
    return face_list

def main():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(app.models)

    source_faces = load_faces(app)

    swapper = get_model("models/inswapper_128.onnx")

    # Initialize webcam video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'));
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    face_helper_wrapper = FaceRestoreHelperWrapper()

    start_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        s = time.time()
        if not ret:
            break

        faces = app.get(frame)
        print("face detection: ", time.time() - s, end="")

        biggest_face = None
        max_area = 0

        height, width, _ = frame.shape

        # find biggest face from faces array.
        for face in faces:
            bbox = face['bbox']

            # bbox can be out of range of frame (for example, jaw is partly out of frame)
            # But if bbox does not have enough face key points, codeformers easily generate deformed image
            # I don't like the output such that teeth is placed on the nose holes.
            # To prevent deformed output, exclude the images which does not meet the requirement
            if bbox[0] < 0 or bbox[1] < 0: continue
            if bbox[2] > width or bbox[3] > height: continue

            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_area = area
                biggest_face = face


        if biggest_face is not None:
            # face.__dict__
            # kps: key points
            # landmark_2d_106
            # landmark_3d_68

            past_time = time.time() - start_time
            idx = int(past_time / 10) % len(source_faces)
            frame = swapper.get(frame, biggest_face, source_faces[idx], paste_back=True)

            # cv2.imwrite("assets/swap.png",frame)

            # high resolution method
            # get the position info of biggest face
            bbox = biggest_face['bbox']
            bbox_up = int(bbox[1])
            bbox_left = int(bbox[0])
            bbox_right = int(bbox[2])
            bbox_down = int(bbox[3])

            horizontal_center = (bbox_left + bbox_right) // 2
            vertical_center = (bbox_up + bbox_down) // 2

            # bbox only contains landmark of the detected face.
            # codeFormers require whole face that has from hair to chin 
            # and suppose the input image should be 512*512 square sized image.
            # So, expand the bbox max 1.5 times and reshape into square if possible.
            square_image_half_size = int( max(bbox_down - bbox_up, bbox_right - bbox_left) * 0.75 )
            square_image_up = max(vertical_center - square_image_half_size, 0)
            square_image_left = max(horizontal_center - square_image_half_size, 0)
            square_image_right = min(horizontal_center + square_image_half_size, width)
            square_image_down = min(vertical_center + square_image_half_size, height)


            cropped_face = frame[square_image_up:square_image_down, square_image_left:square_image_right]
            
            # face image should be 512*512 sized.
            restored_face = face_helper_wrapper.call(
                face_img=cropped_face, 
                fidelity_weight=0.5, 
                has_aligned=True
            )
            restored_resized_face = cv2.resize(restored_face, (square_image_right - square_image_left, square_image_down - square_image_up))

            frame[square_image_up:square_image_down, square_image_left:square_image_right] = restored_resized_face

            # cv2.imwrite("assets/out.png",frame)

            # # 顔の境界ボックスを描画
            # bbox = biggest_face.bbox.astype(int)
            # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # # ランドマークを描画
            # if biggest_face.landmark_2d_106 is not None:
            #     for landmark in biggest_face.landmark_2d_106.astype(int):
            #         cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), 2)

            # if biggest_face.kps is not None:
            #     for kp in biggest_face.kps.astype(int):
            #         cv2.circle(frame, tuple(kp), 5, (255, 0, 0), 10)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        print("swap: ", time.time() - s, end="\n\n")

        # Break the loop on 'q' key press
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()