import numpy as np
from flask import Flask, request, jsonify, render_template

#model modules
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
import os


from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename


with open('class_names.npy', 'rb') as f:
    class_names = np.load(f)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#switch to cpu and load the saved model to do prediction
resnet_transfer = models.resnet50(pretrained = True)

for param in resnet_transfer.parameters():
    param.requires_grad = False

resnet_transfer.fc = nn.Linear(2048,133,bias = True)

fc_parameters = resnet_transfer.fc.parameters()

for param in fc_parameters:
    param.requires_grad = True


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

VGG16 = models.vgg16(pretrained=True)

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def upload_image_file():
    if request.method == 'POST':
      file = request.files['file']
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
      filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

      img = Image.open(request.files['file'].stream).convert("RGB")
      in_transform = transforms.Compose([
                              transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))])

      img_tensor = in_transform(img)[:3,:,:].unsqueeze(0)
      model = resnet_transfer
      model.load_state_dict(torch.load('resnet_transfer.pt','cpu'))
      model.eval()
      output = model(img_tensor)
      _,preds_tensor = torch.max(output,1)
      results = class_names[preds_tensor]

      img_path = filename
      image_ext = cv2.imread(img_path)
      gray = cv2.cvtColor(image_ext, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray)
      #img = cv2.imdecode(np.frombuffer(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
      #faces = face_cascade.detectMultiScale(img)

#dog_detector
      dog = VGG16(img_tensor)
      _, preds_tensor_dog = torch.max(dog, 1)
      dog_index = preds_tensor_dog.item()

      if dog_index >= 151 and dog_index <= 268:
          ans1 = "Dog detected!"
          ans2 = ("Looks like a {}".format(results))
      elif len(faces)>0:
          ans1 = "Hello human!"
          ans2 = ("If you were a dog... You might look like a {}".format(results))
      else:
          ans1 = ("Error! Please upload a dog picture or a human picture")
          ans2 = ""
      #return render_template('prediction.html', filename=filename)

      return render_template('upload.html',filename = filename, value1 = ans1, value2 = ans2)


if __name__ == "__main__":
    app.run(debug=False)
