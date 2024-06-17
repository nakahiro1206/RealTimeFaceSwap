import onnxruntime as ort
import numpy as np
import multiprocessing as mp

def init_session(model_path):
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, providers=EP_list)
    return sess

class PickableInferenceSession: # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = init_session(self.model_path)

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.sess = init_session(self.model_path)

class IOProcess (mp.Process):
    def __init__(self):
        super(IOProcess, self).__init__()
        self.sess = PickableInferenceSession('inswapper_128.onnx')

    def run(self):
        print("calling run")
        print(self.sess.run({}, {
            # 'a': np.zeros((3,4),dtype=np.float32), 
            # 'b': np.zeros((4,3),dtype=np.float32), 
            'target': [[[0,0,0]]],
            "source": [[[0,0,0]]]
            }))
        #print(self.sess)

if __name__ == '__main__':
    mp.set_start_method('spawn') # This is important and MUST be inside the name==main block.
    io_process = IOProcess()
    io_process.start()
    io_process.join()