import cv2
import numpy as np
from multiprocessing import Process, Queue, set_start_method
from insightface.app import FaceAnalysis

def extract_face_from_image(image_path, app):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Input image not found.")
        faces = app.get(image)
        if len(faces) == 0:
            raise ValueError("No face detected in the input image.")
        face = faces[0]
        return face, image
    except:
        print("ERRR")

def face_detection_and_swapping(input_queue, output_queue, target_face, target_image_path):
    try:
        app = FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))

        target_image = cv2.imread(target_image_path)
        if target_image is None:
            raise ValueError("Target image not found.")

        target_face_box = target_face.bbox.astype(int)
        target_face_region = target_image[target_face_box[1]:target_face_box[3], target_face_box[0]:target_face_box[2]]

        while True:
            frame = input_queue.get()
            if frame is None:
                break

            faces = app.get(frame)
            if len(faces) == 0:
                output_queue.put(frame)
                continue

            face = faces[0]

            # 顔の交換処理（簡単な例として顔の矩形領域を交換する）
            x1, y1, x2, y2 = face.bbox.astype(int)

            target_face_resized = cv2.resize(target_face_region, (x2 - x1, y2 - y1))

            frame[y1:y2, x1:x2] = target_face_resized

            output_queue.put(frame)
    except Exception as e:
        print(f"Error in face_detection_and_swapping: {e}")

def main():
    set_start_method('spawn')
    input_image_path = 'Tom_Cruise_avp_2014_4.jpg'  # 入力画像のパスを指定
    input_queue = Queue(maxsize=5)
    output_queue = Queue(maxsize=5)

    try:
        app = FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))

        target_face, _ = extract_face_from_image(input_image_path, app)
    except Exception as e:
        print(f"Error in preparing face analysis or extracting face: {e}")
        return

    process = Process(target=face_detection_and_swapping, args=(input_queue, output_queue, target_face, input_image_path))
    process.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if input_queue.full():
            continue

        input_queue.put(frame)

        if not output_queue.empty():
            output_frame = output_queue.get()
            cv2.imshow('Face Swap', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    input_queue.put(None)
    process.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
