import multiprocessing
import multiprocessing.queues
import time
import sys

def add(start_event, stop_event, process_id, input_queue, output_queue):
    while not stop_event.is_set():
        if start_event.is_set():
            # Your processing logic here
            # print(f"Process {process_id} is processing...")
            input_queue.put(time.time())
            time.sleep(0.1)  # Simulate some work being done
        else:
            print(f"Process {process_id} is waiting for start event...")
            time.sleep(0.5)

def process_function(start_event, stop_event, process_id, input_queue, output_queue):
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

            time.sleep(1)  # Simulate some work being done
        else:
            print(f"Process {process_id} is waiting for start event...")
            time.sleep(0.5)

if __name__ == "__main__":
    print(f"cpu_count: {multiprocessing.cpu_count()}")
    start_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    input_queue = multiprocessing.Queue(maxsize=1000)
    output_queue = multiprocessing.Queue(maxsize=1000)

    from model import Frame, Processor
    F = Frame(1,1,1)
    input_queue.put(F)


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

    # Create and start 4 processes
    processes = []
    for i in range(4):
        process = multiprocessing.Process(target=process_function, args=(start_event, stop_event, i, input_queue, output_queue))
        processes.append(process)
        process.start()

    process = multiprocessing.Process(target=add, args=(start_event, stop_event, 999, input_queue, output_queue))
    processes.append(process)
    process.start()

    # try:

    # Start the processes after some time
    print("Starting processes in 3 seconds...")
    time.sleep(3)
    start_event.set()
    
    # Let the processes run for some time
    # time.sleep(10)
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
    
    # except KeyboardInterrupt:
    #     print("Interrupted by user")
    #     stop_event.set()
    #     for process in processes:
    #         process.join()
