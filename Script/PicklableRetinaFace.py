import onnxruntime as ort
import numpy as np
import multiprocessing as mp
import cv2

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def init_session(model_path):
    # EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    EP_list = ['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=EP_list)
    return session

class PicklableInferenceSession:
    """
    Correspond to retinaface"""
    # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = init_session(self.model_path)

    def run(self, *args):# ここでdetect 処理を行う.
        # net_outs = self.session.run(self.output_names, {self.input_name : blob})

        # def get(self, img, max_num=0):
        # bboxes, kpss = self.det_model.detect(img,
        #                                      max_num=max_num,
        #                                      metric='default')
        # if bboxes.shape[0] == 0:
        #     return []
        # ret = []
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     det_score = bboxes[i, 4]
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     face = Face(bbox=bbox, kps=kps, det_score=det_score)
        #     for taskname, model in self.models.items():
        #         if taskname=='detection':
        #             continue
        #         model.get(img, face)
        #     ret.append(face)
        # return ret
        return self.session.run(*args)

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.session = init_session(self.model_path)

    def prepare(self, ctx_id, **kwargs):
        print(kwargs)
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        # input_size will be None
        print(input_size)
        if input_size is not None:
            # if self.input_size is not None:
            #     print('warning: det_size is already set in detection model, ignore')
            # else:
            self.input_size = input_size
        det_thresh = kwargs.get('det_thresh', None)
        self.det_thresh = det_thresh

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        # ここでsession run.
        net_outs = self.session.run(self.output_names, {self.input_name : blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                #solution-1, c style:
                #anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                #for i in range(height):
                #    anchor_centers[i, :, 1] = i
                #for i in range(width):
                #    anchor_centers[:, i, 0] = i

                #solution-2:
                #ax = np.arange(width, dtype=np.float32)
                #ay = np.arange(height, dtype=np.float32)
                #xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                #anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                #solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                #print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size = None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
            
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        det_img[:new_height, :new_width, :] = resized_img

        # call forward.
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                    det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric=='max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

# class PickableInferenceSession(onnxruntime.InferenceSession): 
#     # This is a wrapper to make the current InferenceSession class pickable.
#     def __init__(self, model_path, **kwargs):
#         super().__init__(model_path, **kwargs)
#         self.model_path = model_path

#     def __getstate__(self):
#         return {'model_path': self.model_path}

#     def __setstate__(self, values):
#         model_path = values['model_path']
#         self.__init__(model_path)

class IOProcess (mp.Process): # ここで画像バイナリを渡す.
    def __init__(self, start_event, stop_event):
        super(IOProcess, self).__init__()
        # ここのパスを見つけたらok
        model_path = '/Users/nakanohiroki/.insightface/models/buffalo_l/det_10g.onnx'
        self.session = PicklableInferenceSession(model_path)
        self.start_event = start_event
        self.stop_event = stop_event

        # prepare
        source_face_path = "source_face.jpg"
        source_face = cv2.imread(source_face_path)
        self.session.prepare(ctx_id=0, det_size=(640, 640), input_size=(640, 640), det_thresh=0.5)
        # def get(self, img, max_num=0):
        # bboxes, kpss = self.det_model.detect(img,
        #                                      max_num=max_num,
        #                                      metric='default')
        # if bboxes.shape[0] == 0:
        #     return []
        # ret = []
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     det_score = bboxes[i, 4]
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     face = Face(bbox=bbox, kps=kps, det_score=det_score)
        #     for taskname, model in self.models.items():
        #         if taskname=='detection':
        #             continue
        #         model.get(img, face)
        #     ret.append(face)
        # return ret
        self.session.detect(source_face) # ERROR!
        exit()

        # FaceAnalysis.prepare = RetinaFace.prepare
        # self.session.prepare
        # self.session.

        # processor.store_source_face("Tom_Cruise_avp_2014_4.jpg")
        # """
        # # Load source face
        # img = cv2.imread(img_path)


    def run(self):
        while not self.stop_event.is_set():
            print("calling run")
            print(self.session)
            # print(self.session.run({}, {
            #     # 'a': np.zeros((3,4),dtype=np.float32), 
            #     # 'b': np.zeros((4,3),dtype=np.float32), 
            #     'target': [[[0,0,0]]],
            #     "source": [[[0,0,0]]]
            #     }))

if __name__ == '__main__':
    mp.set_start_method('spawn') # This is important and MUST be inside the name==main block.
    start_event = mp.Event()
    stop_event = mp.Event()
    cpu_num = 4
    io_process_list = []
    for _ in range(cpu_num):
        io_process = IOProcess(start_event, stop_event)
        print("run", _)
        io_process.start()
        io_process_list.append(io_process)
    stop_event.set()
    for io_process in io_process_list:
        io_process.join()