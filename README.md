# PaddleDetection+OpenVINO实现铝片表面缺陷检测

## 1. 项目展示

<iframe src="//player.bilibili.com/player.html?aid=469481179&bvid=BV13541197kC&cid=729397358&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" style="width:100%;height:800px"> </iframe>

## 2. 项目概述

	本项目主要用以解决工业检测中的铝片表面缺陷检测，现代工业中铝合金、钢材等材料的表面缺陷直接影响到产品最终品质与定价，个别缺陷甚至会影响下一个阶段产品的安全可靠性，所以现在工业生产亟需利用人工智能技术手段将智能设备部署于生产线上。该项目针对这一问题，采用PaddlePaddle框架，搭建SSD模型，将其部署到OpenVINO上，同时搭建前端交互页面，实现自动检测、自动处理，降低次品率，提高生产效率，保障生产安全。

## 3. 技术工具

* 模型训练：
	* PaddlePaddle
   * PaddleDetection
* 模型后端部署（python）：
	* onnxruntime
   * flask
* 前端：
	* html、css、js
   * Vue

## 4. 模型训练
### 4.1 [脚本任务训练（推荐）](https://aistudio.baidu.com/aistudio/clusterprojectdetail/3547319)
### 4.2 [NoteBook训练](https://aistudio.baidu.com/aistudio/projectdetail/3499136)

## 5. 模型转换成onnx


```python
!pip -q install paddle2onnx
!pip -q install onnx
!pip -q install onnxruntime
```


```python
# 使用paddle2onnx将paddle模型格式转化到ONNX模型格式。
!paddle2onnx --model_dir model \
    --model_filename __model__ \
    --params_filename __params__ \
    --save_file model.onnx \
    --opset_version 12 \
    --enable_onnx_checker True
```

## 6. 部署

部署所需要的文件（都已提供）：
* server
* nginx-1.20.2

### 6.1 前端部署

前端已经完成，只需要执行以下命令即可启动前端（本地Windows环境下）

* 在`nginx-1.20.2`目录下`powershell`中，输入命令：`./nginx.exe -c conf/det.conf`
* 显示以下即启动成功

![](https://ai-studio-static-online.cdn.bcebos.com/9e045c2c272c4dbeb5afac75ba5e5e61c8d6efe3112f4a718e5fb8e14e738e5c)

### 6.2 python 后端部署
采用的onnxruntime第三方库运行onnx模型
#### 6.2.1 文件夹内容说明
* server/history：存放历史检测信息
* server/label_list.txt：标签
* server/userinfo.json：用户信息
* server/config.json：配置信息
* server/utils.py：通用程序文件
* server/app.py：主程序文件
* server/model.onnx：onnx模型

#### 6.2.2 utils.py


```python
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
```

#### 6.2.3 app.py


```python
import base64
from onnxruntime import InferenceSession
from PIL import Image
from flask import Flask, request, jsonify
import json
import os
import sys
from datetime import timedelta
import time
import copy
from utils import *

app = Flask(__name__)
app.secret_key = 'secret!'

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


@app.route('/register', methods=['POST'])
def register():
    """
    注册模块
    """
    data = json.loads(request.data)['params']
    username = data['username']
    password = data['password']
    if not os.path.exists(app.config['userinfo']):
        userinfo_dict = {}
    else:
        with open(app.config['userinfo']) as f:
            userinfo_dict = json.load(f)

    # 判断用户是否存在, 用户存在, 注册失败
    if userinfo_dict.get(username, None) is not None:
        return json.dumps({"state": 0})

    # 注册成功
    userinfo_dict[username] = password
    with open(app.config['userinfo'], 'w') as f:
        json.dump(userinfo_dict, f)

    return json.dumps({"state": 1})


@app.route('/login', methods=['POST'])
def login():
    """
    登录模块
    """

    if not os.path.exists(app.config['userinfo']):
        return json.dumps({'state': -1})

    data = json.loads(request.data)['params']
    username = data['username']
    password = data['password']
    with open(app.config['userinfo']) as f:
        userinfo_dict = json.load(f)

    # 判断用户是否存在, 用户不存在
    if userinfo_dict.get(username, None) is None:
        return json.dumps({"state": -1})

    # 判断用户密码是否正确, 用户存在且密码正确
    if userinfo_dict.get(username) == password:
        return json.dumps({"state": 1})

    # 用户存在, 密码错误
    return json.dumps({"state": 0})


@app.route('/plot', methods=['GET'])
def plot():
    """
    读取历史信息进行可视化
    :return:
    """

    results = []

    for idx, file_name in enumerate(os.listdir(app.config['history_images'])):
        with open(os.path.join(app.config['history_infos'], file_name.replace('.jpg', '.json'))) as f:
            info = json.load(f)
        results.append({'id': idx, **info})

    return json.dumps({'history': results})


@app.route('/history', methods=['POST'])
def history():
    """
    读取历史预测数据, 两个文件内的文件名一致, 只后缀不一样
    """

    data = json.loads(request.data)['params']
    pageNum = data['pageNum']
    pageSize = data['pageSize']
    end = pageNum * pageSize
    start = end - pageSize

    history_length = len(os.listdir(app.config['history_images']))
    if end > history_length:
        end = history_length
    if start > history_length:
        start = history_length

    results = []

    for idx, file_name in enumerate(os.listdir(app.config['history_images'])[start: end]):
        image = np.array(Image.open(os.path.join(app.config['history_images'], file_name)))
        image = base64.b64encode(np.array(cv2.imencode('.jpg', image)[1]).tobytes()).decode('utf8')
        with open(os.path.join(app.config['history_infos'], file_name.replace('.jpg', '.json'))) as f:
            info = json.load(f)
        results.append({'id': idx, 'image': image, **info})

    return json.dumps({'history': results})


@app.route('/predict', methods=['POST'])
def infer():
    """
    预测接收到的图片
    """

    image = cv2.imdecode(np.array(bytearray(request.files['file'].stream.read())), cv2.IMREAD_COLOR)
    plt_image = copy.deepcopy(image)

    start_time = time.time()
    # 图片预处理
    image = preprocess(image)
    # 预测
    output = app.config['model'].run(output_names=None, input_feed={'image': image})[0]

    # 处理预测结果
    pred = postprocess(output, plt_image, app.config['label_list'])
    pred['predict_time'] = (time.time() - start_time) * 1000
    # 存储
    num = len(os.listdir(app.config['history_images']))
    cv2.imwrite(app.config['history_images'] + f'/{num}.jpg', pred['image'])
    with open(app.config['history_infos'] + f'/{num}.json', 'w') as f:
        json.dump({'predict_time': pred['predict_time'], 'result': pred['result']}, f)

    image_bytes = base64.b64encode(np.array(cv2.imencode('.jpg', pred['image'])[1]).tobytes()).decode('utf8')

    return jsonify({'state': 1, 'image': image_bytes, 'predict_time': pred['predict_time'], 'result': pred['result']})


if __name__ == '__main__':
    if not os.path.exists('./config.json'):
        sys.exit(1)
    # 读取配置
    with open('./config.json') as f:
        cfg = json.load(f)
    for key in cfg:
        if isinstance(cfg[key], str):
            app.config[key] = cfg[key]
        else:
            for sub_key in cfg[key]:
                app.config[key + '_' + sub_key] = cfg[key][sub_key]

    if not os.path.exists(app.config['history_images']):
        os.makedirs(app.config['history_images'])
        os.mkdir(app.config['history_infos'])

    # 加载模型
    app.config['model'] = InferenceSession(app.config['model_path'])

    # 读取label
    app.config['label_list'] = []
    with open(app.config['label_list_path']) as f:
        for line in f.readlines():
            app.config['label_list'].append(line.strip())

    # 启动服务
    app.run(host=app.config['server_ip'], port=app.config['server_port'], debug=True)
```

#### 6.2.4 启动后端

* 在`server`目录下`cmd`中，输入命令：`python app.py`
* 显示以下即启动成功

![](https://ai-studio-static-online.cdn.bcebos.com/e0f5a07da005486fad8973d78cff9d0cbbdaf9650bdd4114924d59fb82986b60)


## 个人简介

* 菜鸡一枚~，啥都不会，干饭第一！！！
* [我在AI Studio上获得钻石等级，点亮9个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/517701](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/517701)
