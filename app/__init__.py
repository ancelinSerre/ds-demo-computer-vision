import json

import boto3
from botocore.exceptions import NoCredentialsError
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import flash
from flask import url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

# AWS Credentials
# with open(".aws/credentials.json", "r") as f:
#     cred = json.loads(f.read())
# ACCESS_KEY = cred["ACCESS_KEY"]
# SECRET_KEY = cred["SECRET_KEY"]

# Upload folder and supported image formats
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MODEL = load_model("app/static/model/model.h5")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def upload_to_aws(local_file, bucket, s3_file):
#     s3 = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
#     try:
#         s3.upload_file(local_file, bucket, s3_file)
#         print("Upload Successful")
#         return True
#     except FileNotFoundError:
#         print("The file was not found")
#         return False
#     except NoCredentialsError:
#         print("Credentials not available")
#         return False

def preprocess_image(filename):
    img = load_img(filename, target_size=(224, 224))  # Charger l'image
    img = img_to_array(img)  # Convertir en tableau numpy
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
    img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16
    return img

def predict_class(img):
    labels = [
        "affichage sauvage",
        "espace vert",
        "graffitis",
        "problème éclairage public",
        "problème signalisation",
        "problème voirie",
        "stationnement abusif"
    ]
    y = MODEL.predict(img)
    y = y.tolist()[0]
    result = {l:round(x,2) for l, x in zip(labels, y)}
    idx = y.index(max(y))
    return (labels[idx], result[labels[idx]])


def create_app():
    app = Flask(__name__)

    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    @app.route("/")
    def homepage():
        return render_template("homepage.html")

    @app.route("/", methods=["GET", "POST"])
    def upload_file():
        if request.method == "POST":
            # check if the post request has the file part
            if "image" not in request.files:
                flash("No file part")
                return redirect(request.url)
            file = request.files["image"]
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == "":
                flash("No selected file")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Local save
                location = f"app/{app.config['UPLOAD_FOLDER']}/{filename}"
                file.save(location)
                img = preprocess_image(location)
                pred = predict_class(img)
                # Send data to aws s3
                # upload_to_aws(
                #     f"app/{app.config['UPLOAD_FOLDER']}/{filename}", 
                #     "arn:aws:s3:eu-west-3:567378022783:accesspoint/ds-computer-vision-demo", 
                #     "data/"+filename
                # )
                # The fact we upload a new file on S3 should trigger a lambda that would act as an endpoint
                # On the lambda we should instantiate the model and predict the image class
                return render_template("homepage.html", data={"name": pred[0], "value": pred[1]})

        return render_template("homepage.html")

    return app