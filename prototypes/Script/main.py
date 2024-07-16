import multiprocessing as mp
import cv2
import time

from PicklableRetinaFace import PicklableRetinaFace
from PicklableArcFace import PicklableArcFace
from PicklableInswapper import PicklableINswapper
from Face import Face

class IOProcess (mp.Process):
    """
    All the member variable should be picklable!
    """
    def __init__(self, start_event, stop_event, input_queue, output_queue, source_face):
        super(IOProcess, self).__init__()
        # RetinaFace 
        model_path = '/Users/nakanohiroki/.insightface/models/buffalo_l/det_10g.onnx'
        self.session = PicklableRetinaFace(model_path)

        # ArcFace implementation
        model_path = '/Users/nakanohiroki/.insightface/models/buffalo_l/w600k_r50.onnx'
        self.recognition_session = PicklableArcFace(model_path)

        model_path = '/Users/nakanohiroki/pythonWorks/RealTimeFaceSwap/inswapper_128.onnx'
        self.swapper_session = PicklableINswapper(model_path)

        self.start_event = start_event
        self.stop_event = stop_event
        self.input_queue = input_queue
        self.output_queue = output_queue

        # All the members should be dependent on input arguments.
        self.source_face = source_face


        # source_face_path = "source_face.jpg"
        # source_face = cv2.imread(source_face_path)
        # bboxes, kpss = self.session.detect(source_face, max_num=0, metric='default')
        # biggest_face_idx = -1
        # face_area = 0
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        #     if area > face_area:
        #         biggest_face_idx = i
        #         face_area = area

        # assert biggest_face_idx != -1
        # det_score = bboxes[biggest_face_idx, 4]
        # kps = None
        # if kpss is not None:
        #     kps = kpss[biggest_face_idx]
        # face = Face(bbox=bbox, kps=kps, det_score=det_score)
        # self.source_face = face # this expression may cause error

    def set_source_face(self, face):
        self.source_face = face

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while not self.stop_event.is_set():
            if not self.start_event.is_set():
                time.sleep(1)
                continue

            # try:
            ret, frame = self.cap.read()

            # frame = self.input_queue.get(timeout=1)

            bboxes, kpss = self.session.detect(frame, max_num=0, metric='default')
            ret = None
            biggest_face_idx = -1
            face_area = 0
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > face_area:
                    biggest_face_idx = i
                    face_area = area
            if (biggest_face_idx != -1):
                det_score = bboxes[biggest_face_idx, 4]
                kps = None
                if kpss is not None:
                    kps = kpss[biggest_face_idx]
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                
                # recognize here
                self.recognition_session.get(frame, face)
                assert face.embedding is not None

                # swap face.
                # this process is not implemented yet.
                assert self.source_face is not None
                # AttributeError: 'IOProcess' object has no attribute 'swapper_session'??
                # frame = self.swapper_session.get(img=frame, target_face=face, source_face=self.source_face)

            try:
                self.output_queue.put(frame, timeout=1)
            except:
                print(f"output_queue put fail")
            # print(f"queue size is {self.input_queue.qsize()} and result is {ret}")
            # except:
            #     print("get fail")
        self.cap.release()

class CaptureProcess (mp.Process): # ここで画像バイナリを渡す.
    """
    All the member variable should be picklable!
    """
    def __init__(self, start_event, stop_event, input_queue):
        super(CaptureProcess, self).__init__()

        self.start_event = start_event
        self.stop_event = stop_event
        self.input_queue = input_queue

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("ret break")
                break

            try: 
                self.input_queue.put(frame, timeout=1)
            except: 
                print("put failed")
            time.sleep(1/120)
        self.cap.release()
        cv2.destroyAllWindows()
            

if __name__ == '__main__':
    mp.set_start_method('spawn') # This is important and MUST be inside the name==main block.
    start_event = mp.Event()
    stop_event = mp.Event()
    manager = mp.Manager()
    input_queue = manager.Queue(maxsize=10)
    output_queue = manager.Queue(maxsize=10)
    # input_queue = mp.Queue(maxsize=1000)
    # output_queue = mp.Queue(maxsize=1000)

    """prepare source face in advance"""
    source_face_path = "source_face.jpg"
    source_image = cv2.imread(source_face_path)

    """
    detect face in the source image
    I should replace this with my code.
    """
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    face = app.get(source_image)
    source_face = None # face[0]

    cpu_num = 4 
    io_process_list = []
    for _ in range(cpu_num):
        io_process = IOProcess(start_event, stop_event, input_queue, output_queue, source_face)
        io_process_list.append(io_process)
    for io_process in io_process_list:
        io_process.start()
        time.sleep(1/30)

    # cap cannot be picklable
    # capture_process = CaptureProcess(start_event, stop_event, input_queue)
    # io_process_list.append(capture_process)
    # capture_process.start()
    
    start_event.set()
    s = time.time() # should be replaced start event.
    while time.time() - s < 10:
        a = time.time()
        # show
        try:
            frame = output_queue.get(timeout=1)# if 1 cpu -> average 0.07sec
            print(f"get frame: {time.time()-a}")
            a = time.time()
            cv2.imshow("output", frame)
            if cv2.waitKey(15) == ord('q'):
                break
        except:
            print("failed to get processed frame")

        try:
            print(f"end of frame: {time.time()-a}, input_queue size is {input_queue.qsize()}, output_queue size is {output_queue.qsize()}")
        except:
            print(f"end of frame: {time.time()-a}")
        a = time.time()

    stop_event.set()
    print('stop event is set now.')
    time.sleep(1)
    for io_process in io_process_list:
        print(io_process)
        io_process.join()
        print('joined')
    cv2.destroyAllWindows()