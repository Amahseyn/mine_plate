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
def drop_and_recreate_database():
    """
    Drop the existing database (if it exists) and recreate it.
    """
    conn = None
    try:
        # Connect to the default 'postgres' database as a superuser
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        conn.autocommit = True
        cursor = conn.cursor()

        # Drop the existing database
        cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {db_name}").format(db_name=sql.Identifier(DB_NAME)))
        print(f"Database '{DB_NAME}' dropped successfully (if it existed).")

        # Create a fresh database
        cursor.execute(sql.SQL("CREATE DATABASE {db_name}").format(db_name=sql.Identifier(DB_NAME)))
        print(f"Database '{DB_NAME}' created successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def droptable():
    """
    Create the 'cameras' table in the newly created database.
    """
    conn = None
    try:
        # Connect to the new database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'cameras' table
        cursor.execute("""Drop TABLE IF EXISTS plates""")
        conn.commit()
        print("Table droped successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
def create_plates_table():
    """
    Create the 'plates' table in the newly created database.
    """
    conn = None
    try:
        # Connect to the new database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'plates' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plates (
                id SERIAL PRIMARY KEY,
                starttime TEXT NOT NULL,
                endtime TEXT,
                raw_image_path TEXT NOT NULL,
                plate_cropped_image_path TEXT NOT NULL,
                predicted_string TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                sentstart BOOLEAN DEFAULT FALSE,
                sentend BOOLEAN DEFAULT FALSE,
                permit BOOLEAN DEFAULT FALSE
            )
        """)
        conn.commit()
        print("Table 'plates' created successfully with 'sent' and 'valid' columns.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()


def create_mine_info_table():
    """
    Create the 'mine_info' table to store information about mines.
    """
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Create the 'mine_info' table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mine_info (
                mine_id SERIAL PRIMARY KEY,  -- Auto-incrementing ID for each mine
                mine_name VARCHAR(100), 
                cameraid VARCHAR(100),  
                location VARCHAR(100),  -- Location of the mine
                owner_name VARCHAR(100),  -- Owner's name of the mine
                contact_number VARCHAR(15)  -- Contact number for the mine
            )
        """)

        conn.commit()
        print("Table 'mine_info' created successfully.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()



# Example usage
if __name__ == "__main__":
    #droptable()
    drop_and_recreate_database()
    create_plates_table()
    create_tables()
    #create_mine_info_table()
    #add_columns_to_plates_table()
    # create_permits_table()
    #create_mine_info_table()
    #deletesample()
    #create_penalties_table()
    #print_table_columns("plates")