import cv2
import time
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
import os
from FaceRestorationHelperWrapper import FaceRestoreHelperWrapper

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
        for face in faces:
            bbox = face['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_area = area
                biggest_face = face

        height, width, _ = frame.shape

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
            bbox = biggest_face['bbox']
            up = max(0, int(bbox[1]))
            left = max(0, int(bbox[0]))
            right = min(width, int(bbox[2]))
            down = min(height, int(bbox[3]))

            cropped_face = frame[up:down, left:right]
            restored_face = face_helper_wrapper.call(
                face_img=cropped_face, 
                fidelity_weight=0.5, 
                has_aligned=True
            )
            restored_resized_face = cv2.resize(restored_face, (right-left, down-up))

            frame[up:down, left:right] = restored_resized_face

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