import cv2
import multiprocessing
import threading
import time
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis

class Processor:
    def __init__(self) -> None:
        self.FPS = 30
        # self.app = None
        # self.swapper = None
        # self.source_face = None

def frame_processor(input_queue, output_queue, stop_event):
    # start_event.wait()  # スタートイベントを待機
    while True:
        if stop_event.is_set():
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

        if not output_queue.full():
            output_queue.put(frame)

def frame_reader(cap, input_queue, processor, stop_thread_event):
    # start_thread_event.wait()
    while True:
        if stop_thread_event[0]:
            print("frame_reader break")
            break
        ret, frame = cap.read()
        if not ret:
            print("ret break")
            break
        if not input_queue.full():
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

    # Prepare the face analysis and swapper models
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    # processor.app = app
    # processor.swapper = get_model("inswapper_128.onnx")

    # # Load source face
    # img = cv2.imread("Tom_Cruise_avp_2014_4.jpg")
    # face = app.get(img)
    # processor.source_face = face[0]

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
    stop_event = multiprocessing.Event()
    start_thread_event = [False]
    stop_thread_event = [False]
    
    # フレーム変換プロセスの開始
    num_processes = 1
    processes = []
    # for _ in range(num_processes):
    #     # stop_event = multiprocessing.Event()
    #     p = multiprocessing.Process(target=frame_processor, args=(input_queue, output_queue, stop_event))
    #     p.start()
    #     processes.append(p)
    p1 = multiprocessing.Process(target=frame_processor, args=(input_queue, output_queue, stop_event))
    p1.start()
    # p2 = multiprocessing.Process(target=frame_processor, args=(input_queue, output_queue, stop_event))
    # p2.start()
    # p3 = multiprocessing.Process(target=frame_processor, args=(input_queue, output_queue, stop_event))
    # p3.start()
    # p4 = multiprocessing.Process(target=frame_processor, args=(input_queue, output_queue, stop_event))
    # p4.start()

    # フレーム読み込みスレッドの開始
    reader_thread = threading.Thread(target=frame_reader, args=(cap, input_queue, processor, stop_thread_event))
    reader_thread.start()
    print("read_thread start")

    # スタートイベントをセットして全プロセスを開始
    start_event[0] = True
    start_thread_event[0] = True
    print("process start")
    
    s = time.time()
    print("main loop(imshow) start")
    while True:
        # print(input_queue.qsize())
        # print(output_queue.qsize())
        # print("-------")
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
    # stop_event.set()
    # スレッドの終了待機
    stop_thread_event[0] = True # This works well.
    print("stop_thread_event set")

    reader_thread.join()
    print("reader_thread joined")

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    print("destroyed")
    
    
    stop_event.set()
    p1.join()
    # for p in processes:
    #     p.join()
    # p1.join()
    # print("p1 joined")
    # p2.join()
    # print("p2 joined")
    # p3.join()
    # print("p3 joined")
    # p4.join()
    # print("p4 joined")
    # print("sub-process joined")

    

if __name__ == '__main__':
    main()