import cv2
import numpy as np
import tensorflow as tf
from twilio.rest import Client

def load_model(tflite_model_path):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

def preprocess_frame(frame, input_size=(256, 256)):
    # Preprocess the frame for your model
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

from twilio.rest import Client

def send_whatsapp_alert(message, phone_number, twilio_phone_number, account_sid, auth_token):
    # Initialize the Twilio client
    client = Client(account_sid, auth_token)
    
    # Send the message via WhatsApp
    message = client.messages.create(
        body=message,
        from_=f'whatsapp:{twilio_phone_number}',
        to=f'whatsapp:{phone_number}'
    )
    
    print(f"Message sent: {message.sid}")


def main(tflite_model_path, class_indices={0: 'fire', 1: 'non_fire'}, phone_number="", twilio_phone_number="", account_sid="", auth_token=""):
    # Load model
    interpreter, input_details, output_details = load_model(tflite_model_path)

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)
    
    fire_detected = False

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Predict fire or non_fire
        prediction = predict_fire(interpreter, input_details, output_details, frame)
        predicted_label = class_indices[prediction]

        # Set text color based on prediction
        if predicted_label == 'fire':
            text_color = (0, 0, 255)  # Red for fire
            if not fire_detected:
                fire_detected = True
                send_whatsapp_alert("Fire detected!", phone_number, twilio_phone_number, account_sid, auth_token)
        else:
            text_color = (0, 255, 0)  # Green for non_fire
            fire_detected = False

        # Display the resulting frame with prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Prediction: {predicted_label}', (20, 40), font, 1, text_color, 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)

        # Break the loop on Enter key press
        if cv2.waitKey(1) & 0xFF == 13:
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tflite_model_path = 'fire_detection_model.tflite'
    phone_number = '+6282297585750'
    twilio_phone_number = '+14155238886'  
    account_sid = 'ACe48883a5e9a9864cc395e89e8b172c30'
    auth_token = 'e529f10cbb8ec1cd23d7fc045633f18a'
    main(tflite_model_path, phone_number=phone_number, twilio_phone_number=twilio_phone_number, account_sid=account_sid, auth_token=auth_token)