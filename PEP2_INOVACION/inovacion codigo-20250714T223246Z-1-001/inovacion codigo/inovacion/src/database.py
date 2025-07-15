import mysql.connector

database = mysql.connector.connect(
    host="localhost",
    user="root",
    password="ZsefV1234$",
    database="innovacion_y_emprendimiento_2"
)

def get_cursor(buffered=True):
    return database.cursor(buffered=buffered)
