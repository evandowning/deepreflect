import sys
import os
import configparser
from db import get_analyst_id

import psycopg2

def connect_db():
    # Read config file
    config = configparser.ConfigParser()
    config.read('db.cfg')

    dbName = config['db']['name']
    dbUser = config['db']['username']
    dbPass = config['db']['password']

    drEmail = config['user']['email']
    drName = config['user']['name']

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

    return conn

def _main():
    # Connect to database
    conn = connect_db()
    if conn == None:
        return
    cur = conn.cursor()

    with conn:
        # Get analyst id
        analyst_id = get_analyst_id(cur)
        analyst_id = analyst_id[0]

        # Fetch this analyst's reviews
        cur.execute("SELECT f.hash,f.family,f.func_addr,r.ts_start,r.ts_end FROM functions AS f, reviews AS r WHERE r.analyst_id = %s AND r.function_id = f.id", (analyst_id,))
        reviews = cur.fetchall()

        if len(reviews) == 0:
            sys.stdout.write('Analyst has not made any reviews yet.\n')
            return

        sys.stdout.write('Number of function reviews: {0}\n'.format(len(reviews)))

        t = 0

        # Calculate average time spent analyzing functions
        for h,family,func_addr,ts_start,ts_end in reviews:
            diff = ts_end - ts_start
            t += diff.total_seconds()

        sys.stdout.write('Average amount of time spent reviewing functions: {0} seconds\n'.format(t/len(reviews)))

        # Calculate number of functions labeled by reviews
        cur.execute("SELECT DISTINCT f1.hash, f1.family, f1.func_addr "
                    "FROM functions AS f1, functions AS f2, reviews AS r "
                    "WHERE (f1.cid = f2.cid) AND (r.function_id = f2.id) "
                   )
        count = len(set(cur.fetchall()))
        count -= len(reviews)

        sys.stdout.write('Number of functions labeled by reviews: {0}\n'.format(count))

    conn.commit()

    # Close connection
    conn.close()

if __name__ == '__main__':
    _main()
