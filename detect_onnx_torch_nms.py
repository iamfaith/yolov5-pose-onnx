import argparse
import time
from pathlib import Path
import os
import cv2
import numpy as np
import onnxruntime as ort

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_imshow, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
                        classes=None, agnostic=False, multi_label=False,
                        labels=(), kpt_label=False, nc=None, nkpt=None):
    """
    Runs Non-Maximum Suppression (NMS) on inference results using NumPy + OpenCV.
    Returns:
        list of detections, each is (n,6) ndarray per image [x1,y1,x2,y2,conf,cls]
    """
    if nc is None:
        nc = prediction.shape[2] - 5 if not kpt_label else prediction.shape[2] - 56

    xc = prediction[..., 4] > conf_thres  # candidates
    min_wh, max_wh = 2, 4096
    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [np.zeros((0, 6), dtype=np.float32)] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]

        # Convert boxes
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = np.where(x[:, 5:] > conf_thres)
            out = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:
            if not kpt_label:
                j = np.argmax(x[:, 5:], axis=1)
                conf = x[np.arange(len(x)), j + 5]
                mask = conf > conf_thres
                out = np.concatenate((box[mask], conf[mask, None], j[mask, None].astype(np.float32)), 1)
            else:
                kpts = x[:, 6:]
                conf = x[:, 4]
                j = np.argmax(x[:, 5:6], axis=1)
                mask = conf > conf_thres
                out = np.concatenate((box[mask], conf[mask, None], j[mask, None].astype(np.float32), kpts[mask]), 1)

        if classes is not None and out.shape[0]:
            mask = np.isin(out[:, 5].astype(int), classes)
            out = out[mask]

        n = out.shape[0]
        if not n:
            continue
        elif n > max_nms:
            order = np.argsort(-out[:, 4])
            out = out[order[:max_nms]]

        # Prepare for cv2.dnn.NMSBoxes
        boxes = out[:, :4].tolist()
        scores = out[:, 4].tolist()
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)

        if len(indices) > 0:
            indices = indices.flatten()
            if len(indices) > max_det:
                indices = indices[:max_det]
            out = out[indices]
        else:
            out = np.zeros((0, out.shape[1]), dtype=np.float32)

        output[xi] = out

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, kpt_label=False, step=2):
    """
    Rescale coords (xyxy or keypoints) from img1_shape to img0_shape using numpy.
    img1_shape: (height, width)
    coords: numpy array of shape (N, 4) for boxes or (N, K*step) for keypoints
    img0_shape: (height, width)
    ratio_pad: (gain, pad) if provided
    kpt_label: True if coords are keypoints (x,y,v), False if boxes
    step: stride for keypoints (default 2 for x,y)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old/new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)  # (pad_w, pad_h)
    else:
        gain, pad = ratio_pad
    if isinstance(gain, (list, tuple)):
        gain = gain[0]

    if not kpt_label:
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, [0, 2]] /= gain
        coords[:, [1, 3]] /= gain
        clip_coords(coords, img0_shape, step=2)
        # coords[:, 0:4] = np.round(coords[:, 0:4])
    else:
        coords[:, 0::step] -= pad[0]  # x padding
        coords[:, 1::step] -= pad[1]  # y padding
        coords[:, 0::step] /= gain
        coords[:, 1::step] /= gain
        clip_coords(coords, img0_shape, step=step)
        # coords = np.round(coords)
    return coords


def clip_coords(boxes, img_shape, step=2):
    """
    Clip bounding boxes or keypoints to image shape using numpy.
    boxes: numpy array
    img_shape: (height, width)
    step: stride (2 for boxes, 2 or 3 for keypoints)
    """
    boxes[:, 0::step] = np.clip(boxes[:, 0::step], 0, img_shape[1])  # x
    boxes[:, 1::step] = np.clip(boxes[:, 1::step], 0, img_shape[0])  # y
    return boxes



def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = \
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label

    kpt_label = True

    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()

    # Load ONNX model
    session = ort.InferenceSession(weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=32)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=32)

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = img.astype(np.float32) / 255.0
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        # Inference
        t1 = time.time()
        pred = session.run(None, {input_name: img})[0]

        pred = pred.reshape(pred.shape[0], -1, pred.shape[-1])
        
        # Apply NMS (non_max_suppression expects torch tensor, so convert)
        # import torch
        # pred = torch.from_numpy(pred)
  
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms,
                                   kpt_label=kpt_label)
        t2 = time.time()

        # Process detections
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]
            gn = np.array(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                for c in np.unique(det[:, 5]):
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {c}{'s' * (n > 1)}, "

                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        xyxy_arr = np.array(xyxy, dtype=np.float32).reshape(1, 4) 
                        xywh = (xyxy2xywh(xyxy_arr) / gn).reshape(-1).tolist()
                        
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:
                        c = int(cls)
                        label = None if opt.hide_labels else f'{c} {conf:.2f}'
                        kpts = det[det_index, 6:]
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                     line_thickness=opt.line_thickness,
                                     kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / str(c) / f'{p.stem}.jpg', BGR=True)

            print(f'{s}Done. ({t2 - t1:.3f}s)')

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_txt_tidl or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" \
            if save_txt or save_txt_tidl else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolo5s.onnx', help='onnx model path(s)')
    parser.add_argument('--img-size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--source', type=str, default='data/images', help='source')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', action='store_true', help='use keypoint labels')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    detect(opt=opt)
