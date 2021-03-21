import sys
import os
import configparser
import hashlib
import psycopg2

PLUGINDIR_PATH = os.path.abspath(os.path.dirname(__file__))

# Calculate sha256sum of file
def sha256_checksum(bv):
    filename = bv.file.filename
    bytez = open(filename,'rb').read()
    sha256 = hashlib.sha256(bytez).hexdigest()
    return sha256

# Connects to database
def connect_db(bv):
    # Read config file
    config = configparser.ConfigParser()
    config.read(os.path.join(PLUGINDIR_PATH,'db.cfg'))

    dbName = config['db']['name']
    dbUser = config['db']['username']
    dbPass = config['db']['password']

    drEmail = config['user']['email']
    drName = config['user']['name']

    # Get SHA256 value of this file
    hash_val = sha256_checksum(bv)

    conn = None

    # Connect to database
    try:
        conn = psycopg2.connect("dbname='{0}' user='{1}' host='localhost' password='{2}'".format(dbName,dbUser,dbPass))
    except Exception as e:
        sys.stderr.write('{0}\n'.format(str(e)))
        sys.stderr.write('No connection made to db. See log window for details.\n')

    # Add analyst if necessary
    with conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO analysts (username, email, name) VALUES (%s, %s, %s) ON CONFLICT (username) DO NOTHING", (dbUser, drEmail, drName))
    conn.commit()

    return conn,hash_val

# Retrieves analyst id
def get_analyst_id(cur):
    # Read config file
    config = configparser.ConfigParser()
    config.read(os.path.join(PLUGINDIR_PATH,'db.cfg'))

    dbUser = config['db']['username']

    cur.execute("SELECT id FROM analysts WHERE username = %s", (dbUser,))
    analyst_id = cur.fetchall()

    return analyst_id
