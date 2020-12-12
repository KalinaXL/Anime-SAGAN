from resources import ImageApi
from flask import Flask, render_template, request
from flask_restful import Api
from flask_mail import Mail, Message
import requests
import io
from PIL import Image
import base64

app = Flask(__name__)
app.secret_key = 'anime_sagan'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'huynhnguyentlm@gmail.com'
app.config['MAIL_PASSWORD'] = ''
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

prefix = '/api/v1'
path = 'http://localhost:5000' + prefix
api = Api(app = app, prefix = prefix)
api.add_resource(ImageApi, '/animeimage')

mail = Mail(app)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/image', methods = ['GET', 'POST'])
def get_image():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        results = requests.get(f'{path}/animeimage')
        if results:
            image_b64 = results.json()['image']
            image_bytes = image_b64.encode('utf-8')
            image_bytes = base64.b64decode(image_bytes)
            msg = Message('[Anime-SAGAN] Your anime image', sender = 'contact@gmail.com', recipients = [email])
            msg.body = f"Hi {name},\n" + "Sincerely,\nKLXL"
            msg.attach('image.png', 'image/png', image_bytes)
            mail.send(msg)
            print('ok')
    return render_template('image.html')

if __name__ == "__main__":
    app.run(debug = True)