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
