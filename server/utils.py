import cv2
import numpy as np


# 调整图片大小
size = (300, 300)
# 标准化均值、方差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# 检测保留阈值
threshold = 0.5
# 检测框显示
color = [
    (237, 189, 101),
    (0, 0, 255),
    (102, 153, 153),
    (255, 0, 0),
    (9, 255, 0)
]


# 数据预处理
def preprocess(image):
    image = cv2.resize(image.astype('float32'), size) / 255.  # HWC
    for i in range(image.shape[-1]):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    # image = (image - np.array(mean)) / np.array(std)  # HWC
    image = image.transpose((2, 0, 1))  # HWC -> CHW
    image = np.array([image])  # CHW -> NCHW
    return image


# 预测结果数据处理
def postprocess(output, image, label_list):
    pred = {'result': []}
    idx = 0
    for item in output:
        if item[1] < threshold:
            continue
        pred['result'].append({
            "class_name": label_list[int(item[0])],
            "score": float(item[1]),
            "xmin": int(item[2] * image.shape[1]),
            "ymin": int(item[3] * image.shape[0]),
            "xmax": int(item[4] * image.shape[1]),
            "ymax": int(item[5] * image.shape[0])
        })
        cv2.rectangle(image, (pred['result'][idx]['xmin'], pred['result'][idx]['ymin']),
                      (pred['result'][idx]['xmax'], pred['result'][idx]['ymax']), color[int(item[0])], 2)
        cv2.putText(image,
                    '{}.{}?:{}'.format(int(item[0]), pred['result'][idx]['class_name'], pred['result'][idx]['score']),
                    (pred['result'][idx]['xmin'], pred['result'][idx]['ymin']), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1)
        idx += 1

    pred['image'] = image
    return pred
