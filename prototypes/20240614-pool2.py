import multiprocessing
from multiprocessing import Event, Pool
import time
import random
from model import Processor

class TaskProcessor:
    def __init__(self) -> None:
        self.a = 1

    def process_type_1(self, task):
        """Process type 1 task"""
        print(self.a)
        print(f"Type 1 - Processing task: {task}")
        time.sleep(random.uniform(0.5, 2.0))  # Simulate a task taking some time
        return f"Type 1 - Task {task} completed"

    def process_type_2(self, task):
        """Process type 2 task"""
        print(f"Type 2 - Processing task: {task}")
        time.sleep(random.uniform(0.5, 2.0))  # Simulate a task taking some time
        return f"Type 2 - Task {task} completed"

def worker(instance, method_name, task):
    """Worker function to call a method on the given instance"""
    method = getattr(instance, "test")
    return method(task)

def main():
    task_queue_1 = multiprocessing.Queue()
    task_queue_2 = multiprocessing.Queue()
    # processor = TaskProcessor()
    s = time.time()

    with multiprocessing.Pool(processes=1) as pool_1, multiprocessing.Pool(processes=4) as pool_2:
        results_1 = []
        results_2 = []
        processor = Processor()

        start_event = Event()
        stop_event = Event()

        task_id = 0
        while time.time()-s < 10:
            task_id += 1
            task_queue_1.put(task_id)  # Add a new task to the first queue
            task_queue_2.put(task_id)  # Add a new task to the second queue

            # while not task_queue_1.empty():
            while not stop_event.is_set():
                if not task_queue_1.empty():
                    task = task_queue_1.get(timeout=1)
                    result = pool_1.apply_async(worker, (processor, 'process_type_1', task))
                    results_1.append(result)

            while not stop_event.is_set():
                if not task_queue_2.empty():
                    task = task_queue_2.get(timeout=1)
                    result = pool_2.apply_async(worker, (processor, 'process_type_2', task))
                    results_2.append(result)

            for result in results_1[:]:
                if result.ready():
                    print(result.get())
                    results_1.remove(result)

            for result in results_2[:]:
                if result.ready():
                    print(result.get())
                    results_2.remove(result)

            time.sleep(1)  # Wait a bit before adding new tasks
        stop_event.set()

if __name__ == '__main__':
    main()
