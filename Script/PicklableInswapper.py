# https://github.com/microsoft/onnxruntime/issues/7846

import onnxruntime as ort
import onnxruntime
import numpy as np
import multiprocessing as mp
import cv2
import onnx
from onnx import numpy_helper
from skimage import transform as trans
from numpy.linalg import norm as l2norm
from Face import Face

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop2(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M

def init_session(model_path):
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=EP_list)
    return session

class PicklableINswapper: # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = init_session(self.model_path)
        print("init")
        self._init_vars()

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.__dict__.update(values)
        # self.model_path = values['model_path']
        # self.session = init_session(self.model_path)
        # print("set state")
        # self._init_vars()

    # Inswapper implementation
    def _init_vars(self):
    # def _init_vars(self, model_file=None, session=None):
        # self.model_file = model_file
        # self.session = session
        # model = onnx.load(self.model_path) # I should remove this declaremnt
        # graph = model.graph
        # graph = self.session.graph
        self.emap = None # numpy_helper.to_array(graph.initializer[-1])
        print(type(self.emap))
        print(self.emap)
        self.input_mean = 0.0
        self.input_std = 255.0
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
        # if self.session is None:
        #     self.session = onnxruntime.InferenceSession(self.model_file, None)
        inputs = self.session.get_inputs()
        self.input_names = []
        for inp in inputs:
            self.input_names.append(inp.name)
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.output_names = output_names
        assert len(self.output_names)==1
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print('inswapper-shape:', self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])
        print('inswapper initialized!')

    # def forward(self, img, latent):
    #     img = (img - self.input_mean) / self.input_std
    #     pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
    #     return pred

    def get(self, img, target_face, source_face: Face, paste_back=True):
        """required members
        input_size, input_std, input_mean, input_names
        emap, session, 
        output_names"""
        return img
    

        aimg, M = norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        
        normed_embedding = None
        if source_face.embedding is not None:
            normed_embedding = source_face.embedding / l2norm(source_face.embedding)

        latent = normed_embedding.reshape((1,-1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
        #print(latent.shape, latent.dtype, pred.shape)
        img_fake = pred.transpose((0,2,3,1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:,:,::-1]
        if not paste_back:
            return bgr_fake, M
        else:
            target_img = img
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            fake_diff[:2,:] = 0
            fake_diff[-2:,:] = 0
            fake_diff[:,:2] = 0
            fake_diff[:,-2:] = 0
            IM = cv2.invertAffineTransform(M)
            img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
            bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            fake_diff = cv2.warpAffine(fake_diff, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
            img_white[img_white>20] = 255
            fthresh = 10
            fake_diff[fake_diff<fthresh] = 0
            fake_diff[fake_diff>=fthresh] = 255
            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask==255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h*mask_w))
            k = max(mask_size//10, 10)
            #k = max(mask_size//20, 6)
            #k = 6
            kernel = np.ones((k,k),np.uint8)
            img_mask = cv2.erode(img_mask,kernel,iterations = 1)
            kernel = np.ones((2,2),np.uint8)
            fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
            k = max(mask_size//20, 5)
            #k = 3
            #k = 3
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            k = 5
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
            img_mask /= 255
            fake_diff /= 255
            #img_mask = fake_diff
            img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            fake_merged = img_mask * bgr_fake + (1-img_mask) * target_img.astype(np.float32)
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged
        