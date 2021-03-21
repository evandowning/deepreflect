from binaryninja import *

import sys
from .db import connect_db

# Prints information about functions
def print_functions(bv,rows):
    sample_hash = rows[0][0]
    sample_family = rows[0][1]
    sys.stdout.write('{0} | {1}\n'.format('sample hash'.ljust(64),'sample family'.ljust(16)))
    sys.stdout.write('{0} | {1}\n'.format(sample_hash.ljust(64),sample_family.ljust(16)))

    sys.stdout.write('{0} | {1} | {2} | {3}\n'.format('function address'.ljust(18),'function label'.ljust(64),'label type'.ljust(16),'analyst'.ljust(32)))
    for row in rows:
        func_addr = row[2]
        func_label = row[3]
        analyst_username = row[4]
        label_type = row[5]

        # If function exists in this sample
        if bv.get_function_at(int(func_addr,16)) is not None:
            sys.stdout.write('{0} | {1} | {2} | {3}\n'.format(func_addr.ljust(18),func_label.ljust(64),label_type.ljust(16),analyst_username.ljust(32)))
        else:
            sys.stdout.write('{0} | {1} | {2}\n'.format(func_addr.ljust(18),func_label.ljust(64),'Error: function not in binary'))

    sys.stdout.write('\n')

# Get all functions
def get_all_functions(cur,hash_val):
    functions = list()

    # Fetch this sample's functions' manual labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, l.name, a.username, 'manual' "
                "FROM functions AS f, reviews AS r, labels AS l, analysts AS a "
                "WHERE (f.hash = %s) AND (r.function_id = f.id) AND (r.analyst_id = a.id) AND (r.label_id = l.id) "
                ,(hash_val,))
    function_labels = cur.fetchall()

    # Fetch this sample's functions' assumed labels
    cur.execute("SELECT f1.hash, f1.family, f1.func_addr, l.name, ' ', 'cluster' "
                "FROM functions AS f1, functions AS f2, reviews AS r, labels AS l "
                "WHERE (f1.hash = %s) AND (f1.cid = f2.cid) AND (r.function_id = f2.id) AND (r.label_id = l.id) "
                ,(hash_val,))
    function_clusters = cur.fetchall()

    # Fetch this sample's functions' non-labels (no manual or cluster association)
    cur.execute("SELECT f.hash, f.family, f.func_addr, 'unlabeled', ' ', ' ' "
                "FROM functions AS f "
                "WHERE (f.hash = %s) AND NOT EXISTS (SELECT 1 FROM reviews AS r WHERE f.id = r.function_id) AND NOT EXISTS (SELECT 1 FROM functions AS f2, reviews as r WHERE f.cid = f2.cid AND f2.id = r.function_id) "
                ,(hash_val,))
    function_unlabels = cur.fetchall()

    functions = function_labels + function_clusters +  function_unlabels

    if len(functions) == 0:
        show_message_box("File not found in DB", "This file has not yet been analyzed. Please follow steps to run sample through model & cluster RoI's before using this plugin", MessageBoxButtonSet.OKButtonSet, MessageBoxIcon.ErrorIcon)
        return None

    return functions

# Get highlighted functions
def get_highlight_functions(cur,hash_val):
    functions = list()

    # Fetch this sample's functions' labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, l.name, a.username, 'manual' "
                "FROM functions AS f, reviews AS r, labels AS l, analysts AS a "
                "WHERE (f.hash = %s) AND (f.cid != -2) AND (r.function_id = f.id) AND (r.analyst_id = a.id) AND (r.label_id = l.id) "
                ,(hash_val,))
    function_labels = cur.fetchall()

    # Fetch this sample's functions' assumed labels
    cur.execute("SELECT f1.hash, f1.family, f1.func_addr, l.name, ' ', 'cluster' "
                "FROM functions AS f1, functions AS f2, reviews AS r, labels AS l "
                "WHERE (f1.hash = %s) AND (f1.cid != -2) AND (f1.cid = f2.cid) AND (r.function_id = f2.id) AND (r.label_id = l.id) "
                ,(hash_val,))
    function_clusters = cur.fetchall()

    # Fetch this sample's functions' non-labels (no manual or cluster association)
    cur.execute("SELECT f.hash, f.family, f.func_addr, 'unlabeled', ' ', ' ' "
                "FROM functions AS f "
                "WHERE (f.hash = %s) AND (f.cid != -2) AND NOT EXISTS (SELECT 1 FROM reviews AS r WHERE f.id = r.function_id) AND NOT EXISTS (SELECT 1 FROM functions AS f2, reviews as r WHERE f.cid = f2.cid AND f2.id = r.function_id) "
                ,(hash_val,))
    function_unlabels = cur.fetchall()

    functions = function_labels + function_clusters + function_unlabels

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
        print_functions(bv,functions)

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
        print_functions(bv,functions)

    conn.commit()

    # Close connection
    conn.close()

