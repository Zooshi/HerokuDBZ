from flask import Flask, request, jsonify
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
import os
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNet
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = r'pics'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
#os.chdir(r'C:\Users\danie\Desktop\Heroku DBZ Projekt')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model_resnet50.h5')

@app.route('/')
def index():
    return render_template('ImageML.html')


@app.route('/api/image', methods=['POST'])
def upload_image():
    # check if the post request has the file part
    if 'image' not in request.files:
        #return jsonify({'error':'No posted image. Should be attribute named image.'})
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    file = request.files['image']

    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        #return jsonify({'error':'Empty filename submitted.'})
        return render_template('ImageML.html', prediction='You did not select an image')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print("***2:"+filename)
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        x = []
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)
        preds=np.argmax(pred, axis=1)
        if preds==0:
            preds="Hello Freezer"
        elif preds==1:
            preds="Hello Son Goku"
        else:
            preds="Hello Vegeta"

#         items = []
#         for itm in lst[0]:
#             items.append({'name':itm[1],'prob':float(itm[2])})

#         response = {'pred':items}
        response = preds
        return render_template('ImageML.html', prediction='{}'.format(response))
        #return jsonify(response)
    else:
        #return jsonify({'error':'File has invalid extension'})
        return render_template('ImageML.html', prediction='Invalid File extension')

if __name__ == '__main__':
    app.run(debug=True ,use_reloader=False)