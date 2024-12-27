import eventlet
eventlet.monkey_patch()
from flask import Flask, Response, send_file, jsonify,send_from_directory
from psycopg2.extras import RealDictCursor
from readsensor import *
# import socket
# import json
from torchvision import transforms
import base64
from psycopg2 import sql, OperationalError, DatabaseError
# import socket
from flask import request, jsonify, send_file
import psycopg2
import cv2
import threading
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import os
import torch
from configParams import Parameters
import datetime
import psycopg2
import warnings
import requests
warnings.filterwarnings("ignore", category=FutureWarning)

params = Parameters()
from flask_cors import CORS, cross_origin
from datetime import datetime, timedelta
from psycopg2 import sql, OperationalError, DatabaseError
from flask_socketio import SocketIO, emit


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

DB_NAME = "license_plate_db"
DB_USER = "postgres"
DB_PASSWORD = "m102030m"
DB_HOST = "localhost"
DB_PORT = "5432"
video_capture = None
frame = None
lock = threading.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_object = YOLO("weights/best.pt")
modelCharX = torch.hub.load('yolov5', 'custom', "model/CharsYolo.pt", source='local', force_reload=True)

font_path = "vazir.ttf"
persian_font = ImageFont.truetype(font_path, 20)
dirpath = os.getcwd()
images_dir = 'images'
raw_images_dir = os.path.join(images_dir, 'raw')
plate_images_dir = os.path.join(images_dir, 'plate')


def detectPlateChars(croppedPlate):
    """Detect characters on a cropped plate."""
    chars, englishchars, confidences = [], [], []
    results = modelCharX(croppedPlate)
    detections = results.pred[0]
    detections = sorted(detections, key=lambda x: x[0])  # Sort by x coordinate
    clses = []
    for det in detections:
        conf = det[4]
        
        if conf > 0.5:
            cls = int(det[5].item())  # Ensure cls is an integer
            clses.append(int(cls))
            char = params.char_id_dict.get(str(int(cls)), '')  # Get character or empty string
            englishchar = params.char_id_dict1.get(str(int(cls)), '')  # Get English character or empty string
            chars.append(char)
            englishchars.append(englishchar)
            confidences.append(conf.item())
    state= False
    if len(chars)==8:
        if 10<=clses[2]<=42:
            for i in [0,1,3,4,5,6,7]:
                if clses[i]<10:
                    state = True
    return state, chars, englishchars, confidences




last_detection_time = {}
def process_frame(img, cameraId):
    global last_char_display, last_detection_time
    tick = time.time()

    results = model_object(img, conf=0.7, stream=True)

    for detection in results:
        bbox = detection.boxes
        for box in bbox:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            plate_img = img[y1:y2, x1:x2]

            cls_names = int(box.cls[0])
            if cls_names == 1:
                state, chars, englishchars, charConfAvg = detectPlateChars(plate_img)
                char_display = []
                englishchardisplay = []

                if state:
                    for persianchar in chars:
                        char_display.append(persianchar)
                    for englishchar in englishchars:
                        englishchardisplay.append(englishchar)

                    current_char_display = ''.join(englishchardisplay)
                    current_time = datetime.now()

                    if current_char_display in last_detection_time:
                        last_time = last_detection_time[current_char_display]
                        time_diff = (current_time - last_time).total_seconds() / 60

                        if time_diff < 5:
                            persian_output = f"{char_display[0]}{char_display[1]}-{char_display[2]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[6]}{char_display[7]}"
                            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))
                            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            txtout = f"Detected {last_time.strftime('%Y-%m-%d %H:%M:%S')}"
                            cv2.putText(img, txtout, (100, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(10, 50, 255), thickness=2, lineType=cv2.LINE_AA)
                            continue

                    englishoutput = f"{englishchardisplay[0]}{englishchardisplay[1]}-{englishchardisplay[2]}-{englishchardisplay[3]}{englishchardisplay[4]}{englishchardisplay[5]}-{englishchardisplay[6]}{englishchardisplay[7]}"
                    persian_output = f"{char_display[0]}{char_display[1]}-{char_display[2]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[6]}{char_display[7]}"
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
                    raw_filename = f"raw_{timestamp}.jpg"
                    plate_filename = f"plt_{timestamp}.jpg"

                    raw_path = os.path.join('static/images/raw', raw_filename)
                    plate_path = os.path.join('static/images/plate', plate_filename)

                    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                    os.makedirs(os.path.dirname(plate_path), exist_ok=True)

                    cv2.imwrite(raw_path, img)
                    cv2.imwrite(plate_path, plate_img)

                    raw_url = f"http://localhost:5000/static/images/raw/{raw_filename}"
                    plate_url = f"http://localhost:5000/static/images/plate/{plate_filename}"
                    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
                    cursor = conn.cursor()

                    try:
                        # Try to update the endtime if the plate has no endtime
                        cursor.execute(
                            """
                            UPDATE plates
                            SET endtime = %s
                            WHERE predicted_string = %s AND endtime IS NULL
                            RETURNING id, endtime
                            """,
                            (timestamp, englishoutput)
                        )

                        # Fetch the plate ID and endtime after the update
                        plate_data = cursor.fetchone()

                        if plate_data is None:  # No row was updated, so insert a new record
                            cursor.execute(
                                """
                                INSERT INTO plates (starttime, raw_image_path, plate_cropped_image_path, predicted_string, camera_id)
                                VALUES (%s, %s, %s, %s, %s)
                                RETURNING id
                                """,
                                (timestamp, raw_url, plate_url, englishoutput, cameraId)
                            )
                            plate_id = cursor.fetchone()[0]
                            time_field = "starttime"
                        else:
                            plate_id = plate_data[0]
                            endtime = plate_data[1]
                            time_field = "endtime"

                            if endtime == timestamp:  # If endtime matches, send the data via socket
                                data = {
                                    "id": str(plate_id),
                                    "endtime": str(timestamp),
                                    "raw_image_path": str(raw_url),
                                    "plate_cropped_image_path": str(plate_url),
                                    "predicted_string": str(englishoutput)
                                }
                                socketio.emit('plate_detected', data)

                        conn.commit()

                    finally:
                        cursor.close()
                        conn.close()

                    last_detection_time[current_char_display] = current_time

                    # Always send a new detection record if it's a fresh plate detection
                    data = {
                        "id": str(plate_id),
                        time_field: str(timestamp),
                        "raw_image_path": str(raw_url),
                        "plate_cropped_image_path": str(plate_url),
                        "predicted_string": str(englishoutput)
                    }

                    socketio.emit('plate_detected', data)

    tock = time.time()
    elapsed_time = tock - tick
    fps_text = f"FPS: {1 / elapsed_time:.2f}"
    print(fps_text)

    return img



@app.route('/camera/<int:cameraId>/stream', methods=['GET'])
@cross_origin(supports_credentials=True)
def video_feed(cameraId):
    def generate():
        global frame
        conn = None

        try:
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cursor = conn.cursor()

            # Fetch the camera link based on the cameraId
            cursor.execute("SELECT cameralink FROM cameras WHERE id = %s", (cameraId,))
            camera_link = cursor.fetchone()

            if camera_link is None:
                return jsonify({"error": "Camera not found"}), 404

            camera_link = camera_link[0]
            camera_link = "a09.mp4"  # Extract link from tuple
            cap = cv2.VideoCapture(camera_link)
            
            if not cap.isOpened():
                print(f"Failed to open video stream from {camera_link}")
                return jsonify({"error": "Failed to open camera stream"}), 500

            while True:
                ret, img = cap.read()
                if not ret:
                    print("No frames read. Exiting...")
                    break

                img = cv2.resize(img, (1280, 720))
                
                processed_frame = process_frame(img,cameraId)


                with lock:
                    frame = processed_frame

                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.01)

        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 500
        finally:
            if conn:
                cursor.close()
                conn.close()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_db_connection():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    return conn
@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('message')
def handle_message(data):
    print(f"Message received: {data}")
    # Optionally, you can emit a response back
    emit('response', {'status': 'Message received'})
# Function to send data to another server
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
# Function to process images and send data
def process_and_send_data():
    while True:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Fetch unsent records
            cursor.execute("""
                SELECT * 
                FROM plates 
                WHERE (sentstart = FALSE OR (endtime IS NOT NULL AND sentend = FALSE)) 
                LIMIT 10;
            """)
            plates = cursor.fetchall()

            if plates:
                for plate in plates:
                    plate_id, starttime, endtime, raw_image_path, plate_image_path, predicted_string, camera_id, sentstart, sentend = plate
                    raw_image_path = raw_image_path.replace("http://localhost:5000/","")
                    plate_image_path = plate_image_path.replace("http://localhost:5000/","")
                    # Process images to base64
                    raw_image = encode_image_to_base64(raw_image_path)
                    plate_image = encode_image_to_base64(plate_image_path)

                    data = {
                        "plate_id": plate_id,
                        "starttime": starttime,
                        "raw_image": raw_image,
                        "plate_image": plate_image,
                        "predicted_string": predicted_string,
                        "camera_id": camera_id,
                    }

                    if endtime:
                        data["endtime"] = endtime
                    

                    success = False
                    while not success:
                        try:
                            response = requests.post("http://5.10.248.37:5000/sync", json=data)
                            if response.status_code == 200:
                                success = True
                                # Update sent status
                                if endtime:
                                    cursor.execute("UPDATE plates SET sentend = TRUE WHERE id = %s", (plate_id,))
                                else:
                                    cursor.execute("UPDATE plates SET sentstart = TRUE WHERE id = %s", (plate_id,))
                                conn.commit()
                                print(f"Data sent successfully for plate ID {plate_id}")
                            else:
                                print(f"Server error: {response.status_code} for plate ID {plate_id}. Retrying...")
                        except requests.exceptions.RequestException as e:
                            print(f"Connection error: {e}. Retrying in 10 seconds...")
                            time.sleep(10)  # Retry delay
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error in processing and sending data: {e}")

# Start the background thread
def start_background_thread():
    thread = threading.Thread(target=process_and_send_data)
    thread.daemon = True  # Ensure thread stops with the application
    thread.start()


# Function to schedule sending data every 5 minutes
@app.route('/plates', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_all_plates():
    conn = None
    try:
        # Get query parameters
        page = request.args.get('page', type=int, default=1)
        limit = request.args.get('limit', type=int, default=10)

        # Dynamic filters
        filters = []
        params = []

        # Add dynamic filters based on input arguments
        if 'platename' in request.args:
            search_value = request.args.get('platename').lower().replace('-', '')
            filters.append("REPLACE(LOWER(predicted_string), '-', '') LIKE %s")
            params.append(f"%{search_value}%")

        if 'starttime' in request.args:
            filters.append("starttime = %s")
            params.append(request.args.get('starttime'))

        if 'endtime' in request.args:
            filters.append("endtime = %s")
            params.append(request.args.get('endtime'))

        if 'id' in request.args:
            filters.append("id = %s")
            params.append(request.args.get('id', type=int))

        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        # Build the base query
        base_query = """
            SELECT id, starttime,endtime, predicted_string, raw_image_path, plate_cropped_image_path, camera_id
            FROM plates
        """

        # Add WHERE clause if there are filters
        if filters:
            base_query += " WHERE " + " AND ".join(filters)

        # Fetch the total count with filters
        count_query = "SELECT COUNT(*) FROM plates"
        if filters:
            count_query += " WHERE " + " AND ".join(filters)

        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()[0]

        # Handle pagination or fetch all records
        if page == 0:
            # Fetch all records without pagination
            query = base_query + " ORDER BY id DESC"
            cursor.execute(query, tuple(params))
        else:
            # Fetch records with pagination
            offset = (page - 1) * limit
            query = base_query + " ORDER BY id DESC LIMIT %s OFFSET %s"
            cursor.execute(query, tuple(params) + (limit, offset))
        plates = cursor.fetchall()

        # Format the results
        plates_list = []
        for row in plates:
 
            plates_list.append({
                "id": row[0],
                "starttime": row[1],
                "endtime":row[2],
                "predicted_string": row[3],
                "raw_image_path": row[4],
                "cropped_plate_path": row[5]
            })

        # Build the response
        response = {
            "count": total_count,
            "plates": plates_list,
        }

        return jsonify(response), 200

    except psycopg2.OperationalError as db_err:
        return jsonify({"error": f"Database connection failed: {db_err}"}), 500
    except psycopg2.DatabaseError as sql_err:
        return jsonify({"error": f"SQL error: {sql_err}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()



@app.route('/images/<path:filename>')
@cross_origin(supports_credentials=True)
def serve_image(filename):
    print(f"filename:{filename}")
    return send_from_directory('static', filename)

@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        return Response()
    
# def main():
#     camera_id = 1  # Example camera ID

#     video_thread = threading.Thread(target=video_feed, args=(camera_id,))
#     video_thread.start()

#     schedule_data_transfer()
#     try:
#         while True:
#             time.sleep(.05)
#     except KeyboardInterrupt:
#         print("Stopping threads and scheduler.")
#         video_thread.join()

if __name__ == '__main__':
    start_background_thread() 
    socketio.run(app, host='0.0.0.0',debug=True, port=5000)
