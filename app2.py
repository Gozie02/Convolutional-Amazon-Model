import streamlit as st
import tensorflow as tf
import io
from PIL import Image, ImageOps
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img

index_list = {
    "0" : [
        "WATER",
        "water"
    ],
    "1" : [
        "CLOUDY",
        "cloudy"
    ],
    "2" : [
        "CLEAR",
        "clear"
    ],
    "3" : [
        "HAZE",
        "haze"
    ],
    "4" : [
        "PRIMARY",
        "primary"
    ],
    "5" : [
        "ROAD",
        "road"
    ],
    "6" : [
        "CULTIVATION",
        "cultivation"
    ],
    "7" : [
        "CONVENTIONAL_MINE",
        "conventional_mine"
    ],
    "8" : [
        "SLASH_BURN",
        "slash_burn"
    ],
    "9" : [
        "BLOOMING",
        "blooming"
    ],
    "10" : [
        "AGRICULTURE",
        "agriculture"
    ],
    "11" : [
        "SELECTIVE_LOGGING",
        "selective_logging"
    ],
    "12" : [
        "BARE_GROUND",
        "bare_ground"
    ],
    "13" : [
        "BLOW_DOWN",
        "blow_down"
    ],
    "14" : [
        "PARTLY_CLOUDY",
        "partly_cloudy"
    ],
    "15" : [
        "ARTISINAL_MINE",
        "artisinal_mine"
    ],
    "16" : [
        "HABITATION",
        "habitation"
    ]
}


def decode_predictions(preds, top = 17, class_list_path = index_list):
    if len(preds.shape) != 2 or preds.shape[1] != 17:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 17)). '
                         'Found array with shape: ' + str(preds.shape))
    listOfIndexes = index_list
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(listOfIndexes[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key = lambda x: x[2], reverse = True)
        results.append(result)
    return results

def fbeta(y_true , y_pred, beta=2, epsilon=1e-4):
    squared_beta = beta**2

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(tf.cast(y_pred, tf.float32), tf.constant(0.2)), tf.float32)
        
    tp = tf.reduce_sum(y_true * y_pred, axis=1)
    fp = tf.reduce_sum(y_pred, axis=1) - tp
    fn = tf.reduce_sum(y_true, axis=1) - tp
    
    p = tp/(tp+fp+epsilon)   #precision
    r = tp/(tp+fn+epsilon)   #recall
    
    fb = (1+squared_beta)*p*r / (squared_beta*p + r + epsilon)
    return fb

@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model('model3.hdf5', custom_objects = {"fbeta": fbeta})
    return model
with st.spinner('Model is being loaded...'):
    model = tf.keras.models.load_model('model3.hdf5', custom_objects = {"fbeta": fbeta})
    st.write("""# Image Classification""")
    file = st.file_uploader("Upload the image to be classified", type = ["jpg", "png", "jpeg"])
    st.set_option('deprecation.showfileUploaderEncoding', False)

if file is None:
    st.text("Please upload an image")
elif file is not None:
    img = Image.open(file)
    file_name = file.name
    st.image(img, use_column_width = True)
    with open(file_name, "wb") as f:
        f.write(file.getbuffer())

    img2 = load_img(file_name, target_size = (128,128))
    img2 = np.array(img2)
    img2 = img2 / 255.0
    img2 = img2.reshape(1, 128, 128, 3)
    prediction = model.predict(img2)
    pred_class = decode_predictions(prediction, top = 3)
    image_class = str(pred_class[0][0][1])
    score = pred_class[0][0][2]
    for idn, name, likelihood in pred_class[0]:
        st.write("The image is classified as: {} - {:2f}".format(name, likelihood))
    st.write("The top predicted class is", image_class, "with a similarity score of", score)
