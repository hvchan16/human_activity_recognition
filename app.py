from flask import Flask, request, render_template, Response, redirect
from har_main import VideoCamera
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model(
    r'action.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        # get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed', methods=['POST'])
def video_feed():
    resp = Response(gen(VideoCamera().gen_frame()), mimetype='multipart/x-mixed-replace; boundary=frame')
#     return render_template('index.html', resp=resp)
    return resp


@app.route('/close_feed', methods=['POST', 'GET'])
def close_feed():
    resp = Response(gen(VideoCamera().clseWeb()), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('index.html', resp=resp)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/working')
def working():
    return render_template('working.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
