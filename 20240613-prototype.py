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

def capture_and_detect(start_event, stop_event, process_id, input_queue, cap: VideoCapture, processor: Processor):
    while not stop_event.is_set():
        if start_event.is_set():
            # Your processing logic here
            # print(f"Process {process_id} is processing...")
            input_queue.put(time.time())
            time.sleep(processor.get_frame_time())  # Simulate some work being done
        else:
            print(f"Process {process_id} is waiting for start event...")
            time.sleep(processor.get_frame_time())

def process_function(start_event, stop_event, process_id, input_queue, output_queue, processor: Processor):
    while not stop_event.is_set():
        if start_event.is_set():
            # Your processing logic here
            # print(f"Process {process_id} is processing...")
            # try:
            #     print(input_queue.get())
            # except:
            if not input_queue.empty():
                t = input_queue.get()
                print(t)
                output_queue.put(t)
            # else:
            #     print('input queue is empty')
            print(processor.get_frame_time())
            time.sleep(processor.get_frame_time())  # Simulate some work being done
        else:
            print(f"Process {process_id} is waiting for start event...")
            time.sleep(processor.get_frame_time())

if __name__ == "__main__":
    print(f"cpu_count: {multiprocessing.cpu_count()}")
    print("output_queue: prioritize at main.")

    """
    # try:
    #     q.put(item, block=True, timeout=timeout)
    # except Full:
    #     # キューが満杯の場合、古いアイテムを削除してから新しいアイテムを追加する
    #     try:
    #         q.get(block=False)
    #     except Empty:
    #         pass
    #     q.put(item, block=False)
    """

    # frame = cv2.imread("Tom_Cruise_avp_2014_4.jpg")
    # F = processor.detect_biggest_face(frame, -1)
    # processor.swap_face(F)

    start_event = Event()
    stop_event = Event()

    # Manager も謎.
    print("test")
    m = multiprocessing.Manager()
    i = m.Queue()
    # i.get(timeout=1)
    # print("q")
    input_queue = Queue(maxsize=1000)
    output_queue = Queue(maxsize=1000)
    input_queue.get()

    exit()
    """
    (method) def get(
        block: bool = True,
        timeout: float | None = None
    ) -> Any

    (method) def put(
        item: Any,
        block: bool = True,
        timeout: float | None = None
    ) -> Non
    """

    # create manager dict
    ProcessorManager.register('Processor', Processor)
    # ProcessorManager.register('Processor', Processor)
    ProcessorManager.register('VideoCapture', VideoCapture)
    with ProcessorManager() as manager:

        # setup processor
        processor = manager.Processor()
        # method = getattr(processor, "test")
        # print(processor.test(1111))
        # print(processor.FRAME_TIME)
        # print(processor.FRAME_TIME) # processor is not initialized!
        """
        instances = [myClass(i) for i in range(10)]
        with Pool() as p:
            instances = p.map(job, instances)                # Works
            # instances = p.map(lambda x: x.func, instances) # Fails
        """
        processor.store_source_face("Tom_Cruise_avp_2014_4.jpg")

        # setup video capture
        cap = manager.VideoCapture(0)

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
            if not output_queue.empty():
                t = output_queue.get()
                print(f"output: {t}")
        
        # Stop the processes
        print("Stopping processes...")
        stop_event.set()
        
        # Wait for all processes to terminate
        for process in processes:
            process.join()
        print("Successfully joined!")