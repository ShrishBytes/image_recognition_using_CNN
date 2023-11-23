# Import necessary packages
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf
from skimage.io import imsave
from skimage.transform import resize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.applications.inception_resnet_v2 import preprocess_input
from PIL import Image, ImageChops
import logging

# Initialize TensorFlow graph and Flask app
global graph
graph = tf.get_default_graph()
app = Flask(name)
app.secret_key = "hello"

# Define allowed file extensions, load pre-trained model, and set upload folder
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
model = load_model('trained-model.h5')
UPLOAD_FOLDER = 'files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check for the presence of Inception model and load if available, else download and save it
files = [f for f in os.listdir('.') if os.path.isfile(f)]
checkInception = False
for f in files:
    if f == "inception.h5":
        checkInception = True
        inception = load_model('inception.h5', compile=False)
        break
if not checkInception:
    inception = InceptionResNetV2(weights='imagenet', include_top=True)
    inception.save('inception.h5')
inception.graph = graph

# Define category folders mapping categories to folder names
category_folders = {
    'category1': 'folder1',
    'category2': 'folder2',
    # Add more categories and folder names as needed
}

# Function to create embeddings using the InceptionResNetV2 model
def create_inception_embedding(grayscaled_rgb):
    # Resize the input images and preprocess them
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    # Generate embeddings using the InceptionResNetV2 model
    with graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
        return embed

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route handling file upload and processing
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            # If the submitted form contains a URL, process it directly
            url = request.form['url']
            if 'examples' in url:
                color_file = process(url)
                return render_template('index.html', res='static/getty.jpg')
        except:
            logging.exception('')
        # If no file part in the request, flash an error message and redirect
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        # If no filename or selected file, flash an error message and redirect
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If the file is allowed, save it and call the process function
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            category = request.form['category']  # Get the selected category from the form
            color_file = process(filename, category)  # Call the process function with the selected category
            return render_template('index.html', og=color_file[0], res=color_file[1])
    # Render the initial upload page
    return render_template('index.html')

> Udaykumar Herimath 2( Pu Science ):
# Function to process the image
def process(img, category):
    if 'examples' in img:
        im = Image.open(img)
        name = img.split('.')[0].split('/')[-1]
    else:
        im = Image.open('files/' + img)
        name = img.split('.')[0]
    # Resize and process the image
    old_size = im.size
    ratio = float(256) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (256, 256))
    new_im.paste(im, ((256 - new_size[0]) // 2, (256 - new_size[1]) // 2))
    new_im.save(f'static/processed_png/{name}.png', "PNG")
    a = np.array(img_to_array(load_img(f'static/processed_png/{name}.png')))
    a = a.reshape(1, 256, 256, 3)
    color_me_embed = create_inception_embedding(a)
    a = rgb2lab(1.0 / 255 * a)[:, :, :, 0]
    a = a.reshape(a.shape + (1,))
    with graph.as_default():
        output = model.predict([a, color_me_embed])
        output = output * 128
        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:, :, 0] = a[i][:, :, 0]
            cur[:, :, 1:] = output[i]
            colored_folder = category_folders.get(category, 'other')  # Get the corresponding folder for the category
            colored_img_path = f'static/colored_img/{colored_folder}/{name}.png'
            imsave(colored_img_path, lab2rgb(cur))
            trim(Image.open(f'static/processed_png/{name}.png')).save(colored_img_path)
            processed_folder = category_folders.get(category, 'other')  # Get the corresponding folder for the category
            processed_img_path = f'static/processed_png/{processed_folder}/{name}.png'
            trim(Image.open(f'static/colored_img/{name}.png')).save(processed_img_path)
            return (processed_img_path, colored_img_path)

# Function to trim excess background from an image
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

# Run the Flask application
if name == "main":
    app.run(debug=True)
