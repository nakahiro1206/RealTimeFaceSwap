from multiprocessing import Process, Queue, Event
from multiprocessing.managers import BaseManager
import time
from cv2 import VideoCapture, imshow
import multiprocessing
from model import Frame, Processor

# class ProcessorProxy(NamespaceProxy):
#     _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')
class ProcessorManager(BaseManager):
    pass
class CameraManager(BaseManager):
    pass

def capture_and_detect(start_event, stop_event, process_id, input_queue, cap: VideoCapture, processor: Processor):
    while not stop_event.is_set():
        if start_event.is_set():
            # print(processor.__dict__)
            ret, frame = cap.read()
            if not ret:
                print("ret break")
                break
            F = Frame(-1,-1,-1) # this works.
            # print(processor)
            F = processor.detect_biggest_face(frame=frame, captured_time=time.time())
            input_queue.put(F)
        else:
            print(f"Process {process_id} is waiting for start event...")
        time.sleep(processor.get_frame_time())

def process_function(start_event, stop_event, process_id, input_queue, output_queue, processor: Processor):
    while not stop_event.is_set():
        if start_event.is_set():
            # queue operation. avoid using q.qsize()>0
            try:
                t = input_queue.get(timeout=1)
                print(t)
                output_queue.put(t)
            except:
                pass
        else:
            print(f"Process {process_id} is waiting for start event...")
        time.sleep(processor.get_frame_time())

if __name__ == "__main__":
    print(f"cpu_count: {multiprocessing.cpu_count()}")
    print("output_queue: prioritize at main.")

    start_event = Event()
    stop_event = Event()

    # create manager dict
    ProcessorManager.register('Processor', Processor)
    CameraManager.register('VideoCapture', VideoCapture)
    # VideoCapture はmain にあるべき.

    with ProcessorManager() as manager, multiprocessing.Manager() as m, CameraManager() as Cmanager:
        # Manager も謎.
        input_queue = m.Queue(maxsize=1000)
        output_queue = m.Queue(maxsize=1000)

        # setup processor
        processor = manager.Processor()
        processor.store_source_face("Tom_Cruise_avp_2014_4.jpg")

        # setup video capture
        cap = Cmanager.VideoCapture(0)

        # Create and start 4 processes
        processes = []
        for i in range(4):
            process = Process(target=process_function, args=(start_event, stop_event, i, input_queue, output_queue, processor))
            processes.append(process)
            process.start()

        # capture process
        process = Process(target=capture_and_detect, args=(start_event, stop_event, 999, input_queue, cap, processor))
        processes.append(process)
        process.start()

        # Start the processes after some time
        print("Start processes form now")
        start_event.set()
        
        # Let the processes run for some time
        s = time.time()
        while time.time() - s < 10:
            if output_queue.qsize() > 0:
                t = output_queue.get(timeout = 1)
                print(f"output: {t}")
        
        # Stop the processes
        print("Stopping processes...")
        stop_event.set()
        # Give some time for processes to exit gracefully
        time.sleep(1)
        
        # Wait for all processes to terminate
        for process in processes:
            process.join()
        print("Successfully joined!")