import cv2
import multiprocessing
import threading
import time
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis

class Frame:
    def __init__(self, frame, captured_time, face_bbox):
        self.frame = frame
        self.captured_time = captured_time
        self.face_bbox = face_bbox

class Processor:
    def __init__(self) -> None:
        self.FRAME_TIME = 1/30
        self.source_face = None
        print("Lock function is not implemented")

        # Prepare the face analysis and swapper models
        self.detector = FaceAnalysis(name="buffalo_l")
        self.detector.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = get_model("inswapper_128.onnx")        

    def store_source_face(self, img_path) -> None:
        """
        usage
        ---
        processor.store_source_face("Tom_Cruise_avp_2014_4.jpg")
        """
        # Load source face
        img = cv2.imread(img_path)
        s  = time.time()
        faces = self.detector.get(img)
        print(f"load face {time.time()-s}")
        assert len(faces) > 0, "source face image does not includ any face!"
        self.source_face = faces[0]

    def detect_biggest_face(self, frame, captured_time):
        # print("detect start")
        s = time.time()
        faces = self.detector.get(frame)
        # print("face detected")
        max_face = None
        max_area = 0
        for face in faces:
            bbox = face["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_face = face
                max_area = area
        # print(f"detect bigest face {time.time()-s}")
        F = Frame(frame, captured_time, max_face)
        return F

    def swap_face(self, Frame: Frame):
        s = time.time()
        if Frame.face_bbox is not None:
            Frame.frame = self.swapper.get(Frame.frame, Frame.face_bbox, self.source_face, paste_back=True)
            print(f"swap face {time.time()-s}")
            return Frame
        else:
            return Frame
    
    def test(self, task):
        """Process type 1 task"""
        print(self.FRAME_TIME)
        print(f"Type 1 - Processing task: {task}")
        time.sleep(1)  # Simulate a task taking some time
        return f"Type 1 - Task {task} completed"
    
    def get_frame_time(self):
        return self.FRAME_TIME


def frame_processor(input_queue, output_queue, processor, start_event, stop_event):
    # start_event.wait()  # スタートイベントを待機
    while True:
        if stop_event[0]:
            print("frame process break")
            break
        frame = input_queue.get()
        # if frame is None: continue

        # faces = processor.app.get(frame)
        # max_face = None
        # max_area = 0
        # for face in faces:
        #     bbox = face["bbox"]
        #     area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        #     if area > max_area:
        #         max_face = face
        #         max_area = area
        # if max_face is not None:
        #     frame = processor.swapper.get(frame, max_face, processor.source_face, paste_back=True)

        output_queue.put(frame)

def frame_reader(cap, input_queue, processor, start_thread_event, stop_thread_event):
    # start_thread_event.wait()
    while True:
        if stop_thread_event[0]:
            print("frame_reader break")
            break
        ret, frame = cap.read()
        if not ret:
            print("ret break")
            break
        input_queue.put(frame)
        time.sleep(1/processor.FPS)  # キャプチャのスピード調整（必要に応じて調整）.


"""# try:
#     q.put(item, block=True, timeout=timeout)
# except Full:
#     # キューが満杯の場合、古いアイテムを削除してから新しいアイテムを追加する
#     try:
#         q.get(block=False)
#     except Empty:
#         pass
#     q.put(item, block=False)"""

def main():
    processor = Processor()
    processor.store_source_face("Tom_Cruise_avp_2014_4.jpg")
    frame = cv2.imread("Tom_Cruise_avp_2014_4.jpg")
    F = processor.detect_biggest_face(frame, -1)
    processor.swap_face(F)
    exit()


    # process-time * FPS = workers
    cap = cv2.VideoCapture(1)
    
    # キューの作成
    m = multiprocessing.Manager()
    input_queue = m.Queue(maxsize=10)
    output_queue = m.Queue(maxsize=10)

    # 停止イベントの作成
    # start_event = multiprocessing.Event()
    # stop_event = multiprocessing.Event()
    # start_thread_event = threading.Event()
    # stop_thread_event = threading.Event()
    start_event = [False]
    stop_event = [False]
    start_thread_event = [False]
    stop_thread_event = [True]
    
    # フレーム変換プロセスの開始
    num_processes = 4
    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=frame_processor, args=(input_queue, output_queue, processor, start_event, stop_event, ))
        p.start()
        processes.append(p)

    # フレーム読み込みスレッドの開始
    reader_thread = threading.Thread(target=frame_reader, args=(cap, input_queue, processor, start_thread_event, stop_thread_event, ))
    reader_thread.start()
    print("read_thread start")

    # スタートイベントをセットして全プロセスを開始
    start_event[0] = True
    start_thread_event[0] = True
    print("process start")
    
    s = time.time()
    print("main loop(imshow) start")
    while True:
        print(input_queue.qsize())
        print(output_queue.qsize())
        print("-------")
        # print(time.time()-s)
        if not output_queue.empty():
            processed_frame = output_queue.get()
            cv2.imshow('Processed Frame', processed_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                # keeo the screen 25 ms.
                break

        if time.time()-s > 10: # メインスレッドの終了条件.
            print("main loop break")
            break
        time.sleep(1/processor.FPS)  # 表示のスピード調整（必要に応じて調整）-> 0.2
    
    # 停止イベントをセットして全プロセスを終了させる
    stop_event[0] = True
    # スレッドの終了待機
    stop_thread_event[0] = True
    print("stop_thread_event set")

    reader_thread.join()
    print("reader_thread joined")

    for p in processes:
        p.join()
        print("process joined")

    

    print("sub-process joined")

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    print("destroyed")

if __name__ == '__main__':
    main()