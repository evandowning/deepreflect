from binaryninja import *

import configparser
import hashlib
import psycopg2

# Calculate sha256sum of file
def sha256_checksum(filename):
    bytez = open(filename,'rb').read()
    sha256 = hashlib.sha256(bytez).hexdigest()
    return sha256

#TODO
# Displays highlighted functions
def display_RoI(bv, function):
    # Read config file
    config = configparser.ConfigParser()
    config.read('.binaryninja/plugins/deepreflect/db.cfg')

    dbName = config['db']['name']
    dbUser = config['db']['username']
    dbPass = config['db']['password']

    # Get SHA256 value of this file
    fn = bv.file.filename
    hash_val = sha256_checksum(fn)

    # Connect to database
    try:
        conn = psycopg2.connect("dbname='{0}' user='{1}' host='localhost' password='{2}'".format(dbName,dbUser,dbPass))
    except Exception as e:
        print(str(e))
        print("No connection made to db. See log window for details.")
        return

    # Fetch this sample's highlighted functions
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM dr WHERE hash = %s", (hash_val,))
        functions = cur.fetchall()

        if len(functions) == 0:
            show_message_box("File not found in DB", "This file has not yet been analyzed. Please follow steps to run sample through model / cluster RoI's before using this plugin", MessageBoxButtonSet.OKButtonSet, MessageBoxIcon.ErrorIcon)
            return

        else:
            print("This file has been seen in the wild before. Here are some ROI's")
            print("sample hash | sample family | function address | function label")
            for row in functions:
                print(f"{row[1]} | {row[2]} | {row[4]} | {row[3]} ")

    conn.commit()

#TODO
# Modifies function labels
def modify_label(bv, function):
    print('modify label\n')

PluginCommand.register_for_address("DeepReflect\\Show Highlighted Functions", "Outputs all highlighted functions to BinaryNinja console", display_RoI)
PluginCommand.register_for_address("DeepReflect\\Modify Function Label", "Modify function's MITRE label", modify_label)

