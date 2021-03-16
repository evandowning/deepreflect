#!/usr/bin/python3

import sys
import argparse
import configparser
import time
import random

import psycopg2

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='config file', required=True)
    args = parser.parse_args()

    # Store arguments
    cfgFN = args.cfg

    # Read config file
    config = configparser.ConfigParser()
    config.read(cfgFN)
    dbName = config['db']['name']
    dbUser = config['db']['username']
    dbPass = config['db']['password']

    # Connect to database
    try:
        conn = psycopg2.connect("dbname='{0}' user='{1}' host='localhost' password='{2}'".format(dbName,dbUser,dbPass))
    except Exception as e:
        sys.stderr.write('No connection made to db: {0}\n'.format(str(e)))
        sys.exit(1)

    with conn:
        cur = conn.cursor()

        # Fetch all samples and functions highlighted by DeepReflect
        cur.execute("SELECT * FROM dr WHERE cid != -2")
        functions = cur.fetchall()

        if len(functions) == 0:
            sys.stderr.write('No functions identified by DeepReflect\n')
            sys.exit(1)

        #TODO
        for row in functions:
            print(row)

    # Close connect
    conn.close()

if __name__ == '__main__':
    _main()
