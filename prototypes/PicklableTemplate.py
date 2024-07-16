# https://github.com/microsoft/onnxruntime/issues/7846

import onnxruntime as ort
import numpy as np
import multiprocessing as mp
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
import cv2
import time

class PickableInferenceSession: # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self):
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))
        self.app = app

        my_img = cv2.imread("Tom_Cruise_avp_2014_4.jpg")
        my_face = app.get(my_img)
        self.source_face = my_face[0]
        # https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main
        self.swapper = get_model("inswapper_128.onnx")


    # def run(self, *args):
    #     return self.sess.run(*args)

    def __getstate__(self):
        return {'app': self.app, 'swapper': self.swapper}
        # return self.__dict__


    def __setstate__(self, values):
        if self.app is None:
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0, det_size=(640, 640))
            self.app = app

        if self.source_face is None:
            my_img = cv2.imread("Tom_Cruise_avp_2014_4.jpg")
            my_face = app.get(my_img)
            self.source_face = my_face[0]
        # https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main
        if self.swapper is None:
            self.swapper = get_model("inswapper_128.onnx")

    def swap(self, frame):
        faces = self.app.get(frame)

        res = frame.copy()

        for face in faces:
            res = self.swapper.get(res, face, self.source_face, paste_back=True)
        return res

class IOProcess (mp.Process):
    def __init__(self, start_event, stop_event, input_queue, output_queue):
        super(IOProcess, self).__init__()
        self.sess = PickableInferenceSession()

        self.start_event = start_event
        self.stop_event = stop_event
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=1)
                res = self.sess.swap(frame)
                self.output_queue.put(res, timeout=1)
            except:
                pass

if __name__ == '__main__':
    mp.set_start_method('spawn') # This is important and MUST be inside the name==main block.
    start_event = mp.Event()
    stop_event = mp.Event()
    manager = mp.Manager()
    input_queue = manager.Queue(maxsize=1000)
    output_queue = manager.Queue(maxsize=1000)
    cpu_num = 4
    io_process_list = []
    for _ in range(cpu_num):
        io_process = IOProcess(start_event, stop_event, input_queue, output_queue)
        io_process_list.append(io_process)
    for io_process in io_process_list:
        io_process.start()
    
    cap = cv2.VideoCapture(0)
    s = time.time() # should be replaced start event.
    while time.time() - s < 10:
        ret, frame = cap.read()
        if not ret:
            print("ret break")
            break
        try: input_queue.put(frame, timeout=1)
        except: print("put failed")

        try: 
            c = output_queue.get(timeout=1)
            cv2.imshow("out", c)
            cv2.waitKey(1)
        except:
            pass

    # time.sleep(3)
    stop_event.set()
    for io_process in io_process_list:
        io_process.join()

    cap.release()
    cv2.destroyAllWindows()