import psycopg2
from psycopg2 import sql
import datetime
from datetime import datetime, timedelta

# Database configuration

DB_NAME = "client"
DB_USER = "postgres"
DB_PASSWORD = "m102030m"
DB_HOST = "localhost"
DB_PORT = "5432"
def deletesample():
    """
    Create the 'plates' table in the newly created database.
    """
    try:
        # Connect to the new database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM plates
            WHERE id NOT IN (
                SELECT id FROM plates
                ORDER BY id DESC
                LIMIT 20
            )
        """)
        conn.commit()
        print("Table 'plates' deleted successfully with 'sent' and 'valid' columns.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def update_plates_table():
    """
    Update the 'plates' table by renaming the 'date' column to 'start' 
    and adding a new 'end' column.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Rename 'date' column to 'start' and add 'end' column
        cursor.execute("""
            ALTER TABLE plates
            RENAME COLUMN date TO starttime;
        """)
        cursor.execute("""
            ALTER TABLE plates
            ADD COLUMN endtime TEXT;
        """)

        conn.commit()
        print("Table 'plates' updated successfully: renamed 'date' to 'start' and added 'end' column.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def add_columns_to_plates_table():
    """
    Add 'sent' and 'valid' columns to the 'plates' table.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Add 'sent' and 'valid' columns
        cursor.execute("""
            ALTER TABLE plates
            ADD COLUMN IF NOT EXISTS sent BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS valid BOOLEAN DEFAULT FALSE;
        """)
        conn.commit()
        print("Columns 'sent' and 'valid' added successfully to the 'plates' table.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()


def insert_into_mine_info(mine_name, location, owner_name, contact_number):
    """
    Insert a new record into the 'mine_info' table.
    
    Args:
        mine_name (str): Name of the mine.
        location (str): Location of the mine.
        owner_name (str): Owner's name of the mine.
        established_date (str): Established date of the mine in 'YYYY-MM-DD' format.
        contact_number (str): Contact number for the mine.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Insert data into the 'mine_info' table
        cursor.execute("""
            INSERT INTO mine_info (mine_name, location, owner_name, contact_number)
            VALUES (%s, %s, %s, %s)
        """, (mine_name, location, owner_name, contact_number))

        conn.commit()
        print("Data inserted successfully into 'mine_info' table.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

def print_table_columns(table_name):
    """
    Print the column names of a specified table in the PostgreSQL database.
    
    Args:
        table_name (str): The name of the table whose columns are to be printed.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Query to retrieve column names
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table_name,))

        columns = cursor.fetchall()

        if columns:
            print(f"Columns in table '{table_name}':")
            for column in columns:
                print(f"- {column[0]}")
        else:
            print(f"Table '{table_name}' does not exist or has no columns.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()


def insert_vehicle_permit(license_plate, mine_id, start_date, end_date):
    """
    Insert a record into the vehicle_permit table with auto-generated vehicle_id.
    """
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Insert or get the vehicle_id from vehicle_info
        cursor.execute("""
            INSERT INTO vehicle_info (license_plate)
            VALUES (%s)
            ON CONFLICT (license_plate) DO NOTHING
            RETURNING vehicle_id
        """, (license_plate,))
        
        result = cursor.fetchone()
        if result is None:
            # If the license plate already exists, fetch its vehicle_id
            cursor.execute("SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s", (license_plate,))
            vehicle_id = cursor.fetchone()[0]
        else:
            vehicle_id = result[0]

        # Insert into the vehicle_permit table
        cursor.execute("""
            INSERT INTO vehicle_permit (vehicle_id, mine_id, start_date, end_date)
            VALUES (%s, %s, %s, %s)
        """, (vehicle_id, mine_id, start_date, end_date))
        
        conn.commit()
        print(f"Permit record for vehicle_id {vehicle_id} added successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def get_table_names():
    """
    Retrieve the names of all tables in the connected PostgreSQL database.
    """
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()

        # Query to get table names
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = cursor.fetchall()

        # Extract table names
        table_names = [table[0] for table in tables]
        return table_names

    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()
print_table_columns("vehicle_organization")