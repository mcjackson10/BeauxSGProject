# All-in-One Dog Detector and Deterrent Script (Final Version)
#
# This version includes input normalization for improved accuracy and removes
# the debug mode for clean, operational output.

import cv2
import numpy as np
import time
import requests
from picamera2 import Picamera2
from gpiozero import Servo, OutputDevice
from tflite_runtime.interpreter import Interpreter

# --- User-Defined Settings & Credentials ---

# Pushover Notification Credentials
PUSHOVER_USER_KEY = "ug5on3hcov7y64ioemekzk8i7p7y35"
PUSHOVER_API_TOKEN = "arxfj6m4owmuwu37x1m3tm6rqciza7"

# Model and Label File Paths
MODEL_PATH = "model.tflite"
LABEL_PATH = "labelmap.txt"

# --- KEY ACCURACY SETTING ---
# After testing, you can adjust this. Start low and increase if needed.
CONFIDENCE_THRESHOLD = 0.45  # Set to 45%
TARGET_LABEL = 'dog'

# Cooldown Period
NOTIFICATION_COOLDOWN = 30 
last_notification_time = 0

# --- GPIO and Hardware Setup ---
try:
    servo = Servo(18, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
    relay = OutputDevice(17, active_high=True, initial_value=False)
    servo.mid()
    print("GPIO components initialized successfully.")
except Exception as e:
    print(f"Error initializing GPIO components: {e}")
    print("Running in simulation mode. GPIO actions will be printed to the console.")
    servo = None
    relay = None

# --- Function Definitions ---

def send_pushover_notification(message):
    if PUSHOVER_USER_KEY == "YOUR_PUSHOVER_USER_KEY" or PUSHOVER_API_TOKEN == "YOUR_PUSHOVER_API_TOKEN":
        print("Pushover credentials not set. Skipping notification.")
        return
    try:
        payload = {
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": message,
            "title": "Dog Alert!"
        }
        r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=5)
        r.raise_for_status()
        print("Pushover notification sent successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Pushover notification: {e}")

def squirt_sequence():
    print("Executing squirt sequence...")
    if servo and relay:
        servo.value = -0.5
        time.sleep(0.5)
        relay.on()
        time.sleep(0.5)
        relay.off()
        time.sleep(0.5)
        servo.value = 0.5
        time.sleep(0.5)
        relay.on()
        time.sleep(0.5)
        relay.off()
        time.sleep(0.5)
        servo.mid()
    else:
        print("[SIMULATED] Squirt sequence activated.")

def load_labels(path):
    """Loads the labels file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# --- Main Program ---

labels = load_labels(LABEL_PATH)
interpreter = Interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Check if the model is a quantized model
input_type = input_details[0]['dtype']
is_quantized = input_type == np.uint8

picam2 = Picamera2()
# --- FIX: Use a more robust configuration for headless operation ---
config = picam2.create_still_configuration(main={"format": 'RGB888', "size": (640, 480)})
picam2.configure(config)
# --- END FIX ---
picam2.start()
print("Camera initialized. Starting detection loop...")

try:
    while True:
        # This print statement helps confirm the loop is running
        print("Capturing frame...") 
        frame = picam2.capture_array()
        
        # Prepare the frame for the model
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(im_resized, axis=0)

        # Normalize the image data if the model expects float inputs
        if not is_quantized:
             input_data = (np.float32(input_data) - 127.5) / 127.5

        # Perform detection
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # Loop over all detections
        for i in range(len(scores)):
            current_score = scores[i]
            current_class_id = int(classes[i])

            if current_class_id >= 0 and current_class_id < len(labels):
                current_label = labels[current_class_id]

                # --- DEBUG: Print all detected objects and their scores ---
                print(f"Detected '{current_label}' with {current_score:.2%} confidence.")

                if current_score > CONFIDENCE_THRESHOLD and current_label == TARGET_LABEL:
                    current_time = time.time()
                    if (current_time - last_notification_time) > NOTIFICATION_COOLDOWN:
                        print(f"--- DOG DETECTED with {current_score:.2%} confidence! ---")
                        send_pushover_notification("Dog detected in the kitchen!")
                        squirt_sequence()
                        last_notification_time = current_time
                        # Once we find a dog, we can stop checking other objects in this frame
                        break 
        
except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    # Clean up resources
    picam2.stop()
    if servo and relay:
        servo.detach()
        relay.close()
    print("Resources released. Program terminated gracefully.")

