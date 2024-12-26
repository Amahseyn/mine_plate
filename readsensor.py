import serial
import requests

# API credentials and URL
API_KEY = 'YOUR_API_KEY'
BASE_URL = 'https://map.ir/reverse/'

# Function to send reverse geocoding request
def reverse_geocode(lat, lon):
    headers = {
        'x-api-key': API_KEY,
        'content-type': 'application/json'
    }
    params = {
        'lat': lat,
        'lon': lon
    }
    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        location_name = data.get('address', 'No address found')
        return location_name  # Return location name
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "Error retrieving location"

# Function to read from COM3 and return location name
def read_location_from_com3():
    try:
        # Configure serial connection
        ser = serial.Serial(
            port='COM3',
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        if ser.is_open:
            print(f"Connected to {ser.port} at {ser.baudrate} baudrate.")

        # Read and parse serial data
        while True:
            data = ser.readline()
            if data:
                decoded_data = data.decode('utf-8', errors='replace').strip()
                print(f"Received: {decoded_data}")
                
                # Parse latitude and longitude
                try:
                    latitude, longitude = map(float, decoded_data.split(","))
                    location_name = reverse_geocode(latitude, longitude)
                    print(f"Location: {location_name}")
                    return location_name  # Return the location name
                except ValueError:
                    print("Invalid data format. Expected 'latitude,longitude'.")
    except serial.SerialException as e:
        print(f"Error: {e}")
    finally:
        if ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    # Fetch location name
    location = read_location_from_com3()
    print(f"Final Location: {location}")