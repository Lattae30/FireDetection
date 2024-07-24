import cv2
import numpy as np
import tensorflow as tf

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

def main(tflite_model_path, class_indices={0: 'fire', 1: 'non_fire'}):
    # Load model
    interpreter, input_details, output_details = load_model(tflite_model_path)

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

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
        else:
            text_color = (0, 255, 0)  # Green for non_fire

        # Display the resulting frame with prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Prediction: {predicted_label}', (150, 30), font, 1, text_color, 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)

        # Break the loop on Enter key press
        if cv2.waitKey(1) & 0xFF == 13:
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tflite_model_path = 'fire_detection_model.tflite'
    main(tflite_model_path)