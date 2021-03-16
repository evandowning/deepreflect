from binaryninja import *

import sys
from .db import connect_db

# Prints information about functions
def print_functions(rows):
    sample_hash = rows[0][0]
    sample_family = rows[0][1]
    sys.stdout.write('{0} | {1}\n'.format('sample hash'.ljust(64),'sample family'.ljust(16)))
    sys.stdout.write('{0} | {1}\n'.format(sample_hash.ljust(64),sample_family.ljust(16)))

    sys.stdout.write('{0} | {1}\n'.format('function address'.ljust(18),'function label'.ljust(64)))
    for row in rows:
       func_addr = row[2]
       func_label = row[3]

       sys.stdout.write('{0} | {1}\n'.format(func_addr.ljust(18),func_label.ljust(64)))
    sys.stdout.write('\n')

#TODO - print out labels associated with same cluster

# Get all functions
def get_all_functions(cur,hash_val):
    functions = list()

    # Fetch this sample's functions' labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, l.name "
                "FROM functions AS f, reviews AS r, labels AS l "
                "WHERE (f.hash = %s) AND (f.id = r.function_id) AND (l.id = r.label_id) "
                ,(hash_val,))
    function_labels = cur.fetchall()

    # Fetch this sample's functions' non-labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, 'unlabeled' "
                "FROM functions AS f "
                "WHERE (f.hash = %s) AND NOT EXISTS (SELECT 1 FROM reviews AS r WHERE f.id = r.function_id) "
                ,(hash_val,))
    function_unlabels = cur.fetchall()

    functions = function_labels + function_unlabels

    if len(functions) == 0:
        show_message_box("File not found in DB", "This file has not yet been analyzed. Please follow steps to run sample through model & cluster RoI's before using this plugin", MessageBoxButtonSet.OKButtonSet, MessageBoxIcon.ErrorIcon)
        return None

    return functions

# Get highlighted functions
def get_highlight_functions(cur,hash_val):
    functions = list()

    # Fetch this sample's functions' labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, l.name "
                "FROM functions AS f, reviews AS r, labels AS l "
                "WHERE (f.hash = %s) AND (f.cid != -2) AND (f.id = r.function_id) AND (l.id = r.label_id) "
                ,(hash_val,))
    function_labels = cur.fetchall()

    # Fetch this sample's functions' non-labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, 'unlabeled' "
                "FROM functions AS f "
                "WHERE (f.hash = %s) AND (f.cid != -2) AND NOT EXISTS (SELECT 1 FROM reviews AS r WHERE f.id = r.function_id) "
                ,(hash_val,))
    function_unlabels = cur.fetchall()

    functions = function_labels + function_unlabels

    if len(functions) == 0:
        show_message_box("File not found in DB", "This file has not yet been analyzed. Please follow steps to run sample through model & cluster RoI's before using this plugin", MessageBoxButtonSet.OKButtonSet, MessageBoxIcon.ErrorIcon)
        return None

    return functions

# Displays all functions in database
def display_all(bv):
    # Connect to database
    conn,hash_val = connect_db(bv)
    if conn == None:
        return
    cur = conn.cursor()

    sys.stdout.write('\n')
    sys.stdout.write('DeepReflect: All functions:\n')

    with conn:
        # Get all functions in database
        functions = get_all_functions(cur,hash_val)

        # Print function information
        print_functions(functions)

    conn.commit()

    # Close connection
    conn.close()

# Displays highlight functions in database
def display_highlight(bv):
    # Connect to database
    conn,hash_val = connect_db(bv)
    if conn == None:
        return
    cur = conn.cursor()

    sys.stdout.write('\n')
    sys.stdout.write('DeepReflect: Highlighted functions (unordered):\n')

    with conn:
        # Get highlighted functions in database
        functions = get_highlight_functions(cur,hash_val)

        # Print function information
        print_functions(functions)

    conn.commit()

    # Close connection
    conn.close()

