import cv2
import numpy as np
import tensorflow as tf
import firebase_admin
import time
import requests
from firebase_admin import credentials, messaging, storage, db
from firebase_admin.exceptions import FirebaseError
import logging
import os
from datetime import datetime

# initialize the Firebase Admin SDK
cred = credentials.Certificate('fire-detection-c2a4d-firebase-adminsdk-4wezd-7f6b10b991.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'fire-detection-c2a4d.appspot.com', 
    'databaseURL': 'https://fire-detection-c2a4d-default-rtdb.firebaseio.com/' 
})

# telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = '7373126934:AAFZdIgEfF9zx0zGhxgs_nNuMq0O7ewW-P8'
TELEGRAM_CHAT_ID = '-4226630297'

# Configure logging
logging.basicConfig(filename='notifications.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def load_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def preprocess_frame(frame, input_size=(256, 256)):
    img = cv2.resize(frame, input_size)
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    return img

def predict_fire(interpreter, input_details, output_details, frame):
    img = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = int(tf.round(output_data[0][0]).numpy())
    return prediction

def save_frame(frame, directory='detected_images', filename_prefix='fire_detected'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(directory, f'{filename_prefix}_{timestamp}.jpg')
    cv2.imwrite(filename, frame)
    return filename

def upload_to_storage(file_path):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(os.path.basename(file_path))
        blob.upload_from_filename(file_path)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        logging.error('Error uploading image: %s', e)
        return None

def write_to_database(image_url):
    try:
        ref = db.reference('fire_alerts')
        current_time = datetime.now().isoformat()
        alert_data = {
            'timestamp': current_time,
            'image_url': image_url,
            'status': 'fire detected'
        }
        ref.push(alert_data)
        logging.info('Successfully written to database.')
        print('Successfully written to database.')
        return True
    except FirebaseError as e:
        logging.error('Error writing to database: %s', e)
        print('Error writing to database:', e)
        return False

def send_telegram_message(message, image_path=None):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    response = requests.post(url, data=data)
    
    if image_path:
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
        with open(image_path, 'rb') as image_file:
            files = {'photo': image_file}
            data = {'chat_id': TELEGRAM_CHAT_ID}
            response = requests.post(url, files=files, data=data)
    
    return response

def main(tflite_model_path, class_indices={0: 'fire', 1: 'non_fire'}):
    interpreter, input_details, output_details = load_model(tflite_model_path)
    cap = cv2.VideoCapture(0)
    
    fire_detected_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = predict_fire(interpreter, input_details, output_details, frame)
        predicted_label = class_indices[prediction]

        # check for fire and handle notification timing
        if predicted_label == 'fire':
            if fire_detected_time is None:
                fire_detected_time = time.time()
            elif time.time() - fire_detected_time >= 5:
                text_color = (0, 0, 255)  # red for fire
                frame_path = save_frame(frame)
                image_url = upload_to_storage(frame_path)
                if image_url:
                    if write_to_database(image_url):
                        logging.info(f'Fire alert successfully written to database with image URL: {image_url}')
                        send_telegram_message('WARNING FIRE DETECTED!', frame_path)
                    else:
                        logging.error('Failed to write fire alert to database.')
                fire_detected_time = time.time() 
            else:
                text_color = (0, 0, 255)  # red for fire
        else:
            text_color = (0, 255, 0)  # green for non_fire
            fire_detected_time = None  

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Prediction: {predicted_label}', (150, 30), font, 1, text_color, 2, cv2.LINE_AA)

        # warning text if fire detected for more than 10 seconds
        if fire_detected_time and time.time() - fire_detected_time >= 10:
            cv2.putText(frame, 'WARNING: FIRE DETECTED!', (150, 70), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 13:  # enter key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tflite_model_path = 'fire_detection_model.tflite'
    main(tflite_model_path)
