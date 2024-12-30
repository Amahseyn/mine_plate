import eventlet
eventlet.monkey_patch()
from flask import Flask, Response, send_file, jsonify,send_from_directory
from flask import Flask, request, jsonify
import requests
from psycopg2.extras import RealDictCursor
# from readsensor import *
# import socket
# import json
import base64
from psycopg2 import sql, OperationalError, DatabaseError
# import socket
from flask import request, jsonify, send_file
import psycopg2
# import cv2
import threading
# from ultralytics import YOLO
# from PIL import Image, ImageDraw, ImageFont
# import numpy as np
import time
import os
# import torch
from configParams import Parameters
# import datetime
import psycopg2
import warnings
# import requests
warnings.filterwarnings("ignore", category=FutureWarning)
params = Parameters()
from flask_cors import CORS, cross_origin
from datetime import datetime, timedelta
from psycopg2 import sql, OperationalError, DatabaseError
from flask_socketio import SocketIO, emit
import traceback
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

DB_NAME = "server"
DB_USER = "postgres"
DB_PASSWORD = "m102030m"
DB_HOST = "localhost"
DB_PORT = "5432"
video_capture = None
frame = None
lock = threading.Lock()


dirpath = os.getcwd()
images_dir = 'images'
raw_images_dir = os.path.join(images_dir, 'raw')
plate_images_dir = os.path.join(images_dir, 'plate')

@app.route('/sync', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
          # Debugging
        plate_id = data.get('plate_id')
        starttime = data.get('starttime')
        endtime = data.get('endtime')
        predicted_string = data.get('predicted_string')
        camera_id = data.get('camera_id')
        raw_image_base64 = data.get('raw_image')
        plate_image_base64 = data.get('plate_image')
        permit = data.get("permit")
        
        print(f"Parsed values - plate_id: {plate_id}, starttime: {starttime}, camera_id: {camera_id}")  # Debugging

        conn = get_db_connection()
        cursor = conn.cursor()
        print("Database connection established")  # Debugging

        # Generate filenames based on camera_id and starttime
        filename_prefix = f"{camera_id}_{plate_id}_{starttime.replace(':', '-')}"
        raw_filename = f"{filename_prefix}_raw.jpg"
        plate_filename = f"{filename_prefix}_plate.jpg"
        print(f"Generated filenames - raw: {raw_filename}, plate: {plate_filename}")  # Debugging

        # Save raw image to file
        if raw_image_base64:
            raw_path = os.path.join('static/images/raw', raw_filename)
            with open(raw_path, 'wb') as f:
                f.write(base64.b64decode(raw_image_base64))
            raw_url = f"http://localhost:5000/static/images/raw/{raw_filename}"

        # Save plate image to file
        if plate_image_base64:
            plate_path = os.path.join('static/images/plate', plate_filename)
            with open(plate_path, 'wb') as f:
                f.write(base64.b64decode(plate_image_base64))
            plate_url = f"http://localhost:5000/static/images/plate/{plate_filename}"

        if endtime:
            print("Processing endtime logic")  # Debugging
            cursor.execute('''
                SELECT * FROM plates 
                WHERE plateid = %s AND starttime = %s AND camera_id = %s
            ''', (plate_id, starttime, camera_id))
            existing_row = cursor.fetchone()
            print("Existing row check completed")  # Debugging

            if existing_row:
                print("Updating existing row")  # Debugging
                cursor.execute('''
                    UPDATE plates 
                    SET 
                        endtime = %s, 
                        predicted_string = %s, 
                        raw_image_path = %s, 
                        plate_cropped_image_path = %s,
                        permit = %s 
                    WHERE plateid = %s AND starttime = %s AND camera_id = %s
                ''', (
                    endtime, 
                    predicted_string, 
                    raw_url if raw_image_base64 else None, 
                    plate_url if plate_image_base64 else None, 
                    plate_id, 
                    starttime, 
                    camera_id,
                    permit
                ))
            else:
                print("Inserting new row")  # Debugging
                permit = check_vehicle_permit(cursor,predicted_string, camera_id)
                cursor.execute('''
                    INSERT INTO plates (plateid, starttime, endtime, predicted_string, camera_id, raw_image_path, plate_cropped_image_path,permit) 
                    VALUES (%s, %s, %s, %s, %s, %s,%s,%s)
                ''', (
                    plate_id, 
                    starttime, 
                    endtime, 
                    predicted_string, 
                    camera_id, 
                    raw_url if raw_image_base64 else None, 
                    plate_url if plate_image_base64 else None,
                    permit
                ))
        else:
            print("Processing without endtime")  # Debugging
            valid = check_vehicle_permit(cursor,predicted_string, camera_id)
            cursor.execute('''
                INSERT INTO plates (plateid, starttime, predicted_string, camera_id, raw_image_path, plate_cropped_image_path,permit) 
                VALUES (%s, %s, %s, %s, %s, %s,%s)
            ''', (
                plate_id, 
                starttime, 
                predicted_string, 
                camera_id, 
                raw_url if raw_image_base64 else None, 
                plate_url if plate_image_base64 else None,
                valid
            ))

        conn.commit()
        print("Transaction committed")  # Debugging
        cursor.close()
        conn.close()

        return jsonify({"message": "Data received and database updated successfully"}), 200

    except Exception as e:
        print("Error occurred:")
        traceback.print_exc()  # Prints full traceback for debugging
        return jsonify({"error": str(e)}), 500
def check_vehicle_permit(cursor,license_plate, mine_id):
    """
    Check if a vehicle with a specific license plate has a valid permit for a given mine.

    Parameters:
        license_plate (str): The license plate of the vehicle.
        mine_id (int): The ID of the mine.

    Returns:
        bool: True if the permit is valid, False otherwise.
    """
    try:
        cursor.execute("""
            SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s
        """, (license_plate,))
        vehicle_result = cursor.fetchone()

        if not vehicle_result:
            print(f"Vehicle with license plate '{license_plate}' not found.")
            return False
        
        vehicle_id = vehicle_result[0]

        cursor.execute("""
            SELECT start_date, end_date FROM vehicle_permit
            WHERE vehicle_id = %s AND mine_id = %s
        """, (vehicle_id, mine_id))
        permit_result = cursor.fetchone()

        if not permit_result:
            print(f"No permit found for vehicle '{license_plate}' at mine ID '{mine_id}'.")
            return False

        start_date, end_date = permit_result
        current_date = datetime.now().date()

        if datetime.strptime(start_date, "%Y-%m-%d").date() <= current_date <= datetime.strptime(end_date, "%Y-%m-%d").date():
            print(f"Vehicle '{license_plate}' has a valid permit for mine ID '{mine_id}'.")
            return True
        else:
            print(f"Permit for vehicle '{license_plate}' at mine ID '{mine_id}' is not valid.")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False
@app.route('/plates', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_all_plates():
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

        if 'date' in request.args:
            filters.append("date = %s")
            params.append(request.args.get('date'))

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
            SELECT id, starttime,endtime, predicted_string, raw_image_path, plate_cropped_image_path, permit, camera_id
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
            # Fetch mine_name based on camera_id (mine_id)
            camera_id = row[7]
            cursor.execute("SELECT mine_name FROM mine_info WHERE mine_id = %s", (camera_id,))
            mine_result = cursor.fetchone()
            mine_name = mine_result[0] if mine_result else None
            if row[6]==False:
            # Append plate data to the list
                plates_list.append({
                    "id": row[0],
                    "starttime": row[1],
                    "endtime":row[2],
                    "predicted_string": row[3],
                    "raw_image_path": row[4],
                    "cropped_plate_path": row[5],
                    "permit": row[6],
                    "mine_name": mine_name  # Add mine_name to the response
                })
            
            else:
                query = """
                    SELECT owner_name, organization,contact_number
                    FROM vehicle_info
                    WHERE license_plate = %s
                """
                cursor.execute(query,(row[3],))
                (owner_name,organization,contact_number) =cursor.fetchone()
                plates_list.append({
                    "id": row[0],
                    "starttime": row[1],
                    "endtime":row[2],
                    "predicted_string": row[3],
                    "raw_image_path": row[4],
                    "cropped_plate_path": row[5],
                    "permit": row[6],
                    "mine_name": mine_name,
                    "owner_name":owner_name,
                    "organization":organization,
                    "contact_number":contact_number
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
def get_db_connection():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    return conn
@app.route('/daily_traffic', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_daily_traffic():
    try:
        # Get query parameters
        time1 = request.args.get('time1')
        time2 = request.args.get('time2')

        # Validate input dates
        if not time1 or not time2:
            return jsonify({"error": "time1 and time2 are required"}), 400

        # Ensure the input is in the correct format (YYYY-MM-DD)
        try:
            # Convert the input to datetime objects
            time1 = datetime.strptime(time1, "%Y-%m-%d")
            time2 = datetime.strptime(time2, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format. Use 'YYYY-MM-DD'"}), 400

        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        # Create a date series to ensure all dates in the range are included
        date_series_query = """
            SELECT generate_series(%s::date, %s::date, '1 day'::interval) AS traffic_date
        """
        
        cursor.execute(date_series_query, (time1, time2))
        date_series = cursor.fetchall()

        # Query to count the daily traffic using the `starttime` column as text
        query = """
            SELECT 
                SUBSTRING(starttime, 1, 10) AS traffic_date,  -- Extract the date part from the text
                COUNT(*) AS count
            FROM plates
            WHERE SUBSTRING(starttime, 1, 10) BETWEEN %s AND %s
            GROUP BY traffic_date
            ORDER BY traffic_date;
        """
        
        cursor.execute(query, (time1.strftime("%Y-%m-%d"), time2.strftime("%Y-%m-%d")))
        results = cursor.fetchall()

        # Combine the date series with results, ensuring all dates in the range are included
        daily_traffic = []
        result_dates = {row[0]: row[1] for row in results}

        for date in date_series:
            date_str = str(date[0].date())  # Get only the date part
            count = result_dates.get(date_str, 0)  # Default to 0 if date not in results
            daily_traffic.append({"date": date_str, "count": count})

        # Build the response
        response = {
            "start_date": time1.strftime("%Y-%m-%d"),
            "end_date": time2.strftime("%Y-%m-%d"),
            "daily_traffic": daily_traffic
        }

        return jsonify(response), 200

    except OperationalError as db_err:
        return jsonify({"error": f"Database connection failed: {db_err}"}), 500
    except DatabaseError as sql_err:
        return jsonify({"error": f"SQL error: {sql_err}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        # Ensure cursor and connection are closed safely
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

#-------------Vehicle--------------------------------------------
@app.route('/vehicle', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_all_vehicles():
    try:
        # Get query parameters
        page = request.args.get('page', type=int, default=1)
        limit = request.args.get('limit', type=int, default=10)

        # Dynamic filters
        filters = []
        params = []

        # Add dynamic filters based on input arguments
        if 'license_plate' in request.args:
            search_value = request.args.get('license_plate').lower().replace('-', '')
            filters.append("REPLACE(LOWER(license_plate), '-', '') LIKE %s")
            params.append(f"%{search_value}%")

        if 'owner_name' in request.args:
            filters.append("LOWER(owner_name) LIKE %s")
            params.append(f"%{request.args.get('owner_name').lower()}%")

        if 'organization' in request.args:
            filters.append("LOWER(organization) LIKE %s")
            params.append(f"%{request.args.get('organization').lower()}%")

        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Build the base query
        base_query = """
            SELECT vehicle_id, license_plate, owner_name, organization, contact_number, plate_image
            FROM vehicle_info
        """

        # Add WHERE clause if there are filters
        if filters:
            base_query += " WHERE " + " AND ".join(filters)

        # Fetch the total count with filters
        count_query = "SELECT COUNT(*) FROM vehicle_info"
        if filters:
            count_query += " WHERE " + " AND ".join(filters)

        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()[0]

        # Handle pagination or fetch all records
        if page == 0:
            # Fetch all records without pagination
            query = base_query + " ORDER BY vehicle_id DESC"
            cursor.execute(query, tuple(params))
        else:
            # Fetch records with pagination
            offset = (page - 1) * limit
            query = base_query + " ORDER BY vehicle_id LIMIT %s OFFSET %s"
            cursor.execute(query, tuple(params) + (limit, offset))
        vehicles = cursor.fetchall()

        # Format the results
        vehicles_list = []
        for row in vehicles:
            vehicles_list.append({
                "vehicle_id": row[0],
                "license_plate": row[1],
                "owner_name": row[2],
                "organization": row[3],
                "contact_number": row[4],
                "plate_image": row[5],
            })

        # Build the response
        response = {
            "count": total_count,
            "vehicles": vehicles_list,
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

@app.route('/vehicle', methods=['POST'])
def create_vehicle():
    data = request.json
    plate_id = data.get('plate_id')  # Assume plate_id is provided
    owner_name = data.get('owner_name')
    organization = data.get('organization')
    contact_number = data.get('contact_number')

    if not plate_id:
        return jsonify({"error": "plate_id is required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get plate details
        cursor.execute("SELECT predicted_string, raw_image_path FROM plates WHERE predicted_string = %s", (plate_id,))
        plate = cursor.fetchone()

        if not plate:
            return jsonify({"error": "Plate not found"}), 404

        license_plate, plate_image = plate

        # Insert into vehicle_info
        cursor.execute("""
            INSERT INTO vehicle_info (license_plate, owner_name, organization, contact_number, plate_image)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING vehicle_id
        """, (license_plate, owner_name, organization, contact_number, plate_image))

        vehicle_id = cursor.fetchone()[0]
        conn.commit()
        return jsonify({"vehicle_id": vehicle_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/vehicle/<int:vehicle_id>', methods=['PUT'])
def update_vehicle(vehicle_id):
    data = request.json
    owner_name = data.get('owner_name')
    organization = data.get('organization')
    contact_number = data.get('contact_number')

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT license_plate FROM vehicle_info WHERE vehicle_id = %s", (vehicle_id,))
        predicted_string = cursor.fetchone()[0]
        cursor.execute("SELECT raw_image_path FROM plates WHERE predicted_string = %s", (predicted_string,))
        imagepath = cursor.fetchone()[0]
        
        # Update vehicle_info
        cursor.execute("""
            UPDATE vehicle_info
            SET owner_name = %s, organization = %s, contact_number = %s , plate_image=%s
            WHERE vehicle_id = %s
        """, (owner_name, organization, contact_number, imagepath, vehicle_id))

        if cursor.rowcount == 0:
            return jsonify({"error": "Vehicle not found"}), 404

        conn.commit()

        return jsonify({"message": "Vehicle updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/vehicle/<int:vehicle_id>', methods=['DELETE'])
def delete_vehicle(vehicle_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Delete vehicle_info
        cursor.execute("DELETE FROM vehicle_info WHERE vehicle_id = %s", (vehicle_id,))

        if cursor.rowcount == 0:
            return jsonify({"error": "Vehicle not found"}), 404

        conn.commit()
        return jsonify({"message": "Vehicle deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/vehicle/<int:vehicle_id>', methods=['PATCH'])
def patch_vehicle(vehicle_id):
    data = request.json
    fields = []

    # Dynamically build the SQL query for updating specific fields
    if "owner_name" in data:
        fields.append(("owner_name", data["owner_name"]))
    if "organization" in data:
        fields.append(("organization", data["organization"]))
    if "contact_number" in data:
        fields.append(("contact_number", data["contact_number"]))

    if not fields:
        return jsonify({"error": "No fields to update"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT license_plate FROM vehicle_info WHERE vehicle_id = %s", (vehicle_id,))
        predicted_string = cursor.fetchone()[0]
        cursor.execute("SELECT raw_image_path FROM plates WHERE predicted_string = %s", (predicted_string,))
        imagepath = cursor.fetchone()[0]
        
        # Build dynamic query
        set_clause = ", ".join([f"{field} = %s" for field, _ in fields])
        values = [value for _, value in fields] + [vehicle_id]
        cursor.execute(f"""
            UPDATE vehicle_info
            SET plate_image = %s
            WHERE vehicle_id = %s
        """, (imagepath, vehicle_id))
        cursor.execute(f"""
            UPDATE vehicle_info
            SET {set_clause}
            WHERE vehicle_id = %s
        """, values)
        conn.commit()

        if cursor.rowcount == 0:
            return jsonify({"error": "Vehicle not found"}), 404

        return jsonify({"message": "Vehicle updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()
#Mine ---------------------------------------------------------------------------
### POST: Add a new mine record ###
@app.route('/mine', methods=['POST'])
@cross_origin(supports_credentials=True)
def create_mine():
    data = request.json
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO mine_info (mine_name, cameraid, location, owner_name, contact_number)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING mine_id
        """, (data['mine_name'], data['cameraid'], data.get('location'), data.get('owner_name'), data.get('contact_number')))
        mine_id = cursor.fetchone()[0]
        conn.commit()

        return jsonify({"message": "Mine created successfully", "mine_id": mine_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/mine/<int:mine_id>', methods=['PUT'])
@cross_origin(supports_credentials=True)
def update_mine(mine_id):
    data = request.json
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE mine_info
            SET mine_name = %s, cameraid = %s, location = %s, owner_name = %s, contact_number = %s
            WHERE mine_id = %s
        """, (data['mine_name'], data['cameraid'], data.get('location'), data.get('owner_name'), data.get('contact_number'), mine_id))
        conn.commit()

        return jsonify({"message": "Mine updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/mine/<int:mine_id>', methods=['PATCH'])
@cross_origin(supports_credentials=True)
def patch_mine(mine_id):
    try:
        # Extract data from the request
        data = request.get_json()

        # Build the query dynamically based on the provided fields
        fields = []
        values = []
        for key, value in data.items():
            fields.append(f"{key} = %s")
            values.append(value)

        # Ensure at least one field is provided
        if not fields:
            return jsonify({'error': 'No fields to update provided'}), 400

        values.append(mine_id)  # Add mine_id for the WHERE clause
        query = f"UPDATE mine_info SET {', '.join(fields)} WHERE mine_id = %s;"

        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()

        # Execute the query
        cur.execute(query, values)

        # Commit the transaction
        conn.commit()
        cur.close()
        conn.close()

        # Return success response
        return jsonify({'message': 'Mine partially updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/mine', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_mines():
    try:
        # Extract query parameters
        page = int(request.args.get('page', 1))  # Default to page 1 if not provided
        limit = int(request.args.get('limit', 10))  # Default limit is 10
        search = request.args.get('search', '')  # Search query (default empty)

        # If page is 0, return all records
        if page == 0:
            query = """
                SELECT * FROM mine_info
                WHERE mine_name ILIKE %s OR location ILIKE %s OR owner_name ILIKE %s;
            """
            search_term = f"%{search}%"

            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(query, (search_term, search_term, search_term))
            all_records = cur.fetchall()
            count_query = "SELECT COUNT(*) FROM mine_info WHERE mine_name ILIKE %s OR location ILIKE %s OR owner_name ILIKE %s;"
            cur.execute(count_query, (search_term, search_term, search_term))
            total_count = cur.fetchone()['count']

            cur.close()
            conn.close()

            return jsonify({
                'data': all_records,
                'total_count': total_count
            }), 200

        # Otherwise, apply pagination
        offset = (page - 1) * limit
        query = """
            SELECT * FROM mine_info
            WHERE mine_name ILIKE %s OR location ILIKE %s OR owner_name ILIKE %s
            LIMIT %s OFFSET %s;
        """
        count_query = """
            SELECT COUNT(*) FROM mine_info
            WHERE mine_name ILIKE %s OR location ILIKE %s OR owner_name ILIKE %s;
        """
        search_term = f"%{search}%"

        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Execute the queries
        cur.execute(query, (search_term, search_term, search_term, limit, offset))
        records = cur.fetchall()

        cur.execute(count_query, (search_term, search_term, search_term))
        total_count = cur.fetchone()['count']

        cur.close()
        conn.close()

        # Return the data with pagination info
        return jsonify({
            'data': records,
            'total_count': total_count
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

### DELETE: Remove a mine record ###
@app.route('/mine/<int:mine_id>', methods=['DELETE'])
@cross_origin(supports_credentials=True)
def delete_mine(mine_id):
    try:
        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()

        # Delete the record from the mine_info table
        query = "DELETE FROM mine_info WHERE mine_id = %s;"
        cur.execute(query, (mine_id,))

        # Commit the transaction
        conn.commit()
        cur.close()
        conn.close()

        # Return success response
        return jsonify({'message': 'Mine deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400
#Permit----------------------------------------------------------------------------------------------------
@app.route('/permit', methods=['POST'])
@cross_origin(supports_credentials=True)
def add_vehicle_permit():
    try:
        data = request.get_json()
        license_plate = data['license_plate']
        mine_id = data['mine_id']
        start_date = data['start_date']
        end_date = data['end_date']

        conn = get_db_connection()
        cur = conn.cursor()

        # Insert or fetch vehicle_id from vehicle_info
        cur.execute(
            """
            INSERT INTO vehicle_info (license_plate)
            VALUES (%s)
            ON CONFLICT (license_plate) DO NOTHING
            RETURNING vehicle_id;
            """,
            (license_plate,)
        )
        result = cur.fetchone()
        if result is None:
            cur.execute("SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s", (license_plate,))
            vehicle_id = cur.fetchone()[0]
        else:
            vehicle_id = result[0]

        # Insert data into vehicle_permit
        cur.execute(
            """
            INSERT INTO vehicle_permit (vehicle_id, mine_id, start_date, end_date)
            VALUES (%s, %s, %s, %s);
            """,
            (vehicle_id, mine_id, start_date, end_date)
        )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Vehicle permit added successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 400

### PUT: Update a vehicle permit ###
@app.route('/permit/<int:permit_id>', methods=['PUT'])
@cross_origin(supports_credentials=True)
def update_vehicle_permit(permit_id):
    try:
        data = request.get_json()
        license_plate = data['license_plate']
        mine_id = data['mine_id']
        start_date = data['start_date']
        end_date = data['end_date']

        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch the current vehicle_id linked to the permit
        cur.execute(
            """
            SELECT vehicle_id 
            FROM vehicle_permit
            WHERE permit_id = %s;
            """,
            (permit_id,)
        )
        result = cur.fetchone()
        if not result:
            return jsonify({'error': 'Permit ID not found'}), 404

        current_vehicle_id = result[0]

        # Check if the new license_plate exists in vehicle_info, or add it
        cur.execute(
            """
            INSERT INTO vehicle_info (license_plate)
            VALUES (%s)
            ON CONFLICT (license_plate) DO NOTHING
            RETURNING vehicle_id;
            """,
            (license_plate,)
        )
        result = cur.fetchone()
        if result:
            new_vehicle_id = result[0]
        else:
            # Fetch the vehicle_id for the existing license_plate
            cur.execute("SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s", (license_plate,))
            new_vehicle_id = cur.fetchone()[0]

        # Update the vehicle_permit with the new vehicle_id and other details
        cur.execute(
            """
            UPDATE vehicle_permit
            SET vehicle_id = %s, mine_id = %s, start_date = %s, end_date = %s
            WHERE permit_id = %s;
            """,
            (new_vehicle_id, mine_id, start_date, end_date, permit_id)
        )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Vehicle permit updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

### PATCH: Partially update a vehicle permit ###
@app.route('/permit/<int:permit_id>', methods=['PATCH'])
@cross_origin(supports_credentials=True)
def patch_vehicle_permit(permit_id):
    try:
        data = request.get_json()

        conn = get_db_connection()
        cur = conn.cursor()

        # Check if license_plate is in the data
        if 'license_plate' in data:
            license_plate = data.pop('license_plate')

            # Insert or fetch vehicle_id from vehicle_info
            cur.execute(
                """
                INSERT INTO vehicle_info (license_plate)
                VALUES (%s)
                ON CONFLICT (license_plate) DO NOTHING
                RETURNING vehicle_id;
                """,
                (license_plate,)
            )
            result = cur.fetchone()
            if result:
                vehicle_id = result[0]
            else:
                # Fetch the vehicle_id for the existing license_plate
                cur.execute("SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s", (license_plate,))
                vehicle_id = cur.fetchone()[0]

            # Add vehicle_id to the update fields
            data['vehicle_id'] = vehicle_id

        # Dynamically build the query based on provided fields
        fields = []
        values = []
        for key, value in data.items():
            fields.append(f"{key} = %s")
            values.append(value)

        values.append(permit_id)
        query = f"UPDATE vehicle_permit SET {', '.join(fields)} WHERE permit_id = %s;"

        cur.execute(query, tuple(values))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Vehicle permit updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


### DELETE: Remove a vehicle permit ###
@app.route('/permit/<int:permit_id>', methods=['DELETE'])
@cross_origin(supports_credentials=True)
def delete_vehicle_permit(permit_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Delete the vehicle_permit
        cur.execute("DELETE FROM vehicle_permit WHERE permit_id = %s;", (permit_id,))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Vehicle permit deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/permit', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_vehicle_permits():
    try:
        # Get query parameters
        page = request.args.get('page', type=int, default=1)
        limit = request.args.get('limit', type=int, default=10)
        license_plate = request.args.get('license_plate', '').lower().replace('-', '')

        # Dynamic filters
        filters = []
        params = []

        # Add filter for license_plate if provided
        if license_plate:
            filters.append("REPLACE(LOWER(vi.license_plate), '-', '') LIKE %s")
            params.append(f"%{license_plate}%")

        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Base query to fetch permits with mine_name
        base_query = """
            SELECT 
                vp.*, 
                vi.license_plate, 
                mi.mine_id, 
                mi.mine_name
            FROM 
                vehicle_permit vp
            JOIN 
                vehicle_info vi ON vp.vehicle_id = vi.vehicle_id
            JOIN 
                mine_info mi ON vp.mine_id = mi.mine_id
        """

        # Build WHERE clause if filters are present
        if filters:
            base_query += " WHERE " + " AND ".join(filters)

        # Fetch the total count with filters
        count_query = """
            SELECT COUNT(*)
            FROM 
                vehicle_permit vp
            JOIN 
                vehicle_info vi ON vp.vehicle_id = vi.vehicle_id
            JOIN 
                mine_info mi ON vp.mine_id = mi.mine_id
        """
        if filters:
            count_query += " WHERE " + " AND ".join(filters)

        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()['count']

        # Handle pagination or fetch all records
        if page == 0:
            # Fetch all records without pagination
            query = base_query + " ORDER BY vp.permit_id DESC"
            cursor.execute(query, tuple(params))
        else:
            # Fetch records with pagination
            offset = (page - 1) * limit
            query = base_query + " ORDER BY vp.permit_id DESC LIMIT %s OFFSET %s"
            cursor.execute(query, tuple(params) + (limit, offset))

        records = cursor.fetchall()

        # Close database connection
        cursor.close()
        conn.close()

        # Return response
        return jsonify({
            'data': records,
            'total_count': total_count,
            'page': page,
            'limit': limit if page != 0 else total_count
        }), 200

    except psycopg2.OperationalError as db_err:
        return jsonify({'error': f'Database connection failed: {db_err}'}), 500
    except psycopg2.DatabaseError as sql_err:
        return jsonify({'error': f'SQL error: {sql_err}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500
#--------Organization----------------------------------
@app.route('/organizations', methods=['GET'])
def get_organizations():
    page = request.args.get('page', default=0, type=int)
    limit = request.args.get('limit', default=10, type=int)

    conn = get_db_connection()
    cursor = conn.cursor()

    if page == 0:
        cursor.execute("SELECT * FROM organization")
        organizations = cursor.fetchall()
        count = len(organizations)
    else:
        offset = (page - 1) * limit
        cursor.execute("SELECT * FROM organization LIMIT %s OFFSET %s", (limit, offset))
        organizations = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM organization")
        count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    response = {
        "count": count,
        "organizations": [
            {
                "organization_id": org[0],
                "organization_name": org[1]
            } for org in organizations
        ]
    }
    return jsonify(response), 200

# GET a specific organization by ID
@app.route('/organizations/<int:organization_id>', methods=['GET'])
def get_organization(organization_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM organization WHERE organization_id = %s", (organization_id,))
    organization = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if organization:
        response = {
            "organization_id": organization[0],
            "organization_name": organization[1]
        }
        return jsonify(response), 200
    else:
        return jsonify({"error": "Organization not found"}), 404

# POST (create a new organization)
@app.route('/organizations', methods=['POST'])
def create_organization():
    data = request.json
    organization_name = data.get("organization_name")
    if not organization_name:
        return jsonify({"error": "Organization name is required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO organization (organization_name) VALUES (%s) RETURNING organization_id",
            (organization_name,)
        )
        organization_id = cursor.fetchone()[0]
        conn.commit()
        response = {
            "organization_id": organization_id,
            "organization_name": organization_name
        }
        return jsonify(response), 201
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "Organization name must be unique"}), 400
    finally:
        cursor.close()
        conn.close()

# PATCH (update an organization)
@app.route('/organizations/<int:organization_id>', methods=['PATCH'])
def update_organization(organization_id):
    data = request.json
    organization_name = data.get("organization_name")

    if not organization_name:
        return jsonify({"error": "Organization name is required for update"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE organization SET organization_name = %s WHERE organization_id = %s RETURNING organization_id",
            (organization_name, organization_id)
        )
        updated_id = cursor.fetchone()
        conn.commit()
        if updated_id:
            return jsonify({"organization_id": organization_id, "organization_name": organization_name}), 200
        else:
            return jsonify({"error": "Organization not found"}), 404
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "Organization name must be unique"}), 400
    finally:
        cursor.close()
        conn.close()

# DELETE an organization
@app.route('/organizations/<int:organization_id>', methods=['DELETE'])
def delete_organization(organization_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM organization WHERE organization_id = %s RETURNING organization_id", (organization_id,))
    deleted_id = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()

    if deleted_id:
        return jsonify({"message": "Organization deleted successfully"}), 200
    else:
        return jsonify({"error": "Organization not found"}), 404

# Update Server Database with clent Database 
def process_and_send_data(serveraddress):
    while True:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Fetch unsent records from 'organization', 'vehicle_info', 'vehicle_organization', and 'vehicle_permit'
            cursor.execute("""
                SELECT organization_id, organization_name, sent
                FROM organization
                WHERE sent = FALSE
                LIMIT 10;
            """)
            organizations = cursor.fetchall()

            if organizations:
                for organization in organizations:
                    org_id, org_name, sent = organization

                    # Prepare data for sending
                    data = {
                        "organization_id": org_id,
                        "organization_name": org_name,
                    }

                    success = False
                    while not success:
                        try:
                            # Send data to external server
                            url = f"http://{serveraddress}/sync/organization"
                            response = requests.post(url, json=data)
                            
                            if response.status_code == 200:
                                success = True
                                # Update sent status for organization
                                cursor.execute("UPDATE organization SET sent = TRUE WHERE organization_id = %s", (org_id,))
                                conn.commit()
                                print(f"Organization data sent successfully for ID {org_id}")
                            else:
                                print(f"Server error: {response.status_code} for organization ID {org_id}. Retrying in 1 minute...")
                                time.sleep(60)  # Retry after 1 minute
                        except requests.exceptions.RequestException as e:
                            print(f"Connection error: {e}. Retrying in 1 minute...")
                            time.sleep(60)  # Retry after 1 minute

            # Fetch unsent records from 'vehicle_info'
            cursor.execute("""
                SELECT vehicle_id, license_plate, owner_name, contact_number, plate_image, sent
                FROM vehicle_info
                WHERE sent = FALSE
                LIMIT 10;
            """)
            vehicles = cursor.fetchall()

            if vehicles:
                for vehicle in vehicles:
                    vehicle_id, license_plate, owner_name, contact_number, plate_image, sent = vehicle

                    # Encode plate image to base64
                    #encoded_plate_image = encode_image_to_base64(plate_image)
                    
                    # Prepare data for sending
                    data = {
                        "vehicle_id": vehicle_id,
                        "license_plate": license_plate,
                        "owner_name": owner_name,
                        "contact_number": contact_number,
                        
                    }

                    success = False
                    while not success:
                        try:
                            url = f"http://{serveraddress}/sync/vehicle"
                            # Send data to external server
                            response = requests.post(url, json=data)
                            
                            if response.status_code == 200:
                                success = True
                                # Update sent status for vehicle
                                cursor.execute("UPDATE vehicle_info SET sent = TRUE WHERE vehicle_id = %s", (vehicle_id,))
                                conn.commit()
                                print(f"Vehicle data sent successfully for ID {vehicle_id}")
                            else:
                                print(f"Server error: {response.status_code} for vehicle ID {vehicle_id}. Retrying in 1 minute...")
                                time.sleep(60)  # Retry after 1 minute
                        except requests.exceptions.RequestException as e:
                            print(f"Connection error: {e}. Retrying in 1 minute...")
                            time.sleep(60)  # Retry after 1 minute

            # Fetch unsent records from 'vehicle_organization'
            cursor.execute("""
                SELECT vehicle_id, organization_id, sent
                FROM vehicle_organization
                WHERE sent = FALSE
                LIMIT 10;
            """)
            vehicle_organizations = cursor.fetchall()

            if vehicle_organizations:
                for vehicle_org in vehicle_organizations:
                    vehicle_id, organization_id, sent = vehicle_org

                    # Prepare data for sending
                    data = {
                        "vehicle_id": vehicle_id,
                        "organization_id": organization_id,
                    }

                    success = False
                    while not success:
                        try:
                            url = f"http://{serveraddress}vehicle_organization"
                            response = requests.post(url, json=data)                            
                            if response.status_code == 200:
                                success = True
                                cursor.execute("UPDATE vehicle_organization SET sent = TRUE WHERE vehicle_id = %s AND organization_id = %s", (vehicle_id, organization_id))
                                conn.commit()
                                print(f"Vehicle organization data sent successfully for vehicle ID {vehicle_id} and organization ID {organization_id}")
                            else:
                                print(f"Server error: {response.status_code} for vehicle ID {vehicle_id} and organization ID {organization_id}. Retrying in 1 minute...")
                                time.sleep(60)  # Retry after 1 minute
                        except requests.exceptions.RequestException as e:
                            print(f"Connection error: {e}. Retrying in 1 minute...")
                            time.sleep(60)  # Retry after 1 minute

            # Fetch unsent records from 'vehicle_permit'
            cursor.execute("""
                SELECT permit_id, vehicle_id, mine_id, start_date, end_date, sent
                FROM vehicle_permit
                WHERE sent = FALSE
                LIMIT 10;
            """)
            vehicle_permits = cursor.fetchall()

            if vehicle_permits:
                for permit in vehicle_permits:
                    permit_id, vehicle_id, mine_id, start_date, end_date, sent = permit

                    # Prepare data for sending
                    data = {
                        "permit_id": permit_id,
                        "vehicle_id": vehicle_id,
                        "mine_id": mine_id,
                        "start_date": start_date,
                        "end_date": end_date,
                    }

                    success = False
                    while not success:
                        try:
                            url = f"http://{serveraddress}vehicle_permit"
                            # Send data to external server
                            response = requests.post(url, json=data)
                            
                            if response.status_code == 200:
                                success = True
                                # Update sent status for vehicle_permit
                                cursor.execute("UPDATE vehicle_permit SET sent = TRUE WHERE permit_id = %s", (permit_id,))
                                conn.commit()
                                print(f"Vehicle permit data sent successfully for permit ID {permit_id}")
                            else:
                                print(f"Server error: {response.status_code} for permit ID {permit_id}. Retrying in 1 minute...")
                                time.sleep(60)  # Retry after 1 minute
                        except requests.exceptions.RequestException as e:
                            print(f"Connection error: {e}. Retrying in 1 minute...")
                            time.sleep(60)  # Retry after 1 minute

            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error in processing and sending data: {e}")
            time.sleep(60)  # Retry after 1 minute if there is any issue

def run_in_background(serveraddress):
    thread = threading.Thread(target=process_and_send_data(serveraddress))
    thread.daemon = True  # Ensure the thread is killed when the main program exits
    thread.start()

@app.route('/images/<path:filename>')
@cross_origin(supports_credentials=True)
def serve_image(filename):
    print(f"filename:{filename}")
    return send_from_directory('static', filename)

@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        return Response()
    
if __name__ == '__main__':
    serveraddress = "79.142.76.187"
    run_in_background(serveraddress)
    socketio.run(app, host='0.0.0.0',debug=True, port=5000)
