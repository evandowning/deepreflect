from binaryninja import *

import sys
from .db import connect_db

# Print sorted functions by score
def print_sort(bv,sort_func):
    sample_hash = sort_func[0][0]
    sample_family = sort_func[0][1]
    sys.stdout.write('{0} | {1}\n'.format('sample hash'.ljust(64),'sample family'.ljust(16)))
    sys.stdout.write('{0} | {1}\n'.format(sample_hash.ljust(64),sample_family.ljust(16)))

    # Sort functions by number of callees
    sys.stdout.write('{0} | {1} | {2} | {3} | {4}\n'.format('function address'.ljust(18),'function label'.ljust(64),'label type'.ljust(16),'analyst'.ljust(32),'ranking value'))
    for sample_hash,sample_family,func_addr,func_label,score,analyst_username,label_type in sorted(sort_func, key=lambda x: x[4], reverse=True):

        # If function exists in this sample
        if bv.get_function_at(int(func_addr,16)) is not None:
            sys.stdout.write('{0} | {1} | {2} | {3} | {4}\n'.format(func_addr.ljust(18),func_label.ljust(64),label_type.ljust(16),analyst_username.ljust(32),score))
        else:
            sys.stdout.write('{0} | {1}\n'.format(func_addr.ljust(18),'Error: function not in binary'))

    sys.stdout.write('\n')

# Get highlighted functions
def get_highlight_functions(cur,hash_val):
    functions = list()

    # Fetch this sample's functions' labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, l.name, f.score, a.username, 'manual' "
                "FROM functions AS f, reviews AS r, labels AS l, analysts AS a "
                "WHERE (f.hash = %s) AND (f.cid != -2) AND (r.function_id = f.id) AND (r.analyst_id = a.id) AND (r.label_id = l.id) "
                ,(hash_val,))
    function_labels = cur.fetchall()

    # Fetch this sample's functions' assumed labels
    cur.execute("SELECT f1.hash, f1.family, f1.func_addr, l.name, f1.score, ' ', 'cluster' "
                "FROM functions AS f1, functions AS f2, reviews AS r, labels AS l "
                "WHERE (f1.hash = %s) AND (f1.cid != -2) AND (f1.cid = f2.cid) AND (r.function_id = f2.id) AND (r.label_id = l.id) "
                ,(hash_val,))
    function_clusters = cur.fetchall()

    # Fetch this sample's functions' non-labels (no manual or cluster association)
    cur.execute("SELECT f.hash, f.family, f.func_addr, 'unlabeled', f.score, ' ', ' ' "
                "FROM functions AS f "
                "WHERE (f.hash = %s) AND (f.cid != -2) AND NOT EXISTS (SELECT 1 FROM reviews AS r WHERE f.id = r.function_id) AND NOT EXISTS (SELECT 1 FROM functions AS f2, reviews as r WHERE f.cid = f2.cid AND f2.id = r.function_id) "
                ,(hash_val,))
    function_unlabels = cur.fetchall()

    functions = function_labels + function_clusters + function_unlabels

    if len(functions) == 0:
        show_message_box("File not found in DB", "This file has not yet been analyzed. Please follow steps to run sample through model & cluster RoI's before using this plugin", MessageBoxButtonSet.OKButtonSet, MessageBoxIcon.ErrorIcon)
        return None

    return functions

# Sort functions by DeepReflect score
def sort_score(bv):
    # Connect to database
    conn,hash_val = connect_db(bv)
    if conn == None:
        return
    cur = conn.cursor()

    sys.stdout.write('\n')
    sys.stdout.write('DeepReflect: Highlighted functions sorted by score:\n')

    sort_func = list()

    with conn:
        # Get highlighted functions in database
        functions = get_highlight_functions(cur,hash_val)

        for row in functions:
            sample_hash = row[0]
            sample_family = row[1]
            func_addr = row[2]
            func_label = row[3]
            score = row[4]
            analyst_username = row[5]
            label_type = row[6]

            sort_func.append((sample_hash,sample_family,func_addr,func_label,score,analyst_username,label_type))

    conn.commit()

    # Close connection
    conn.close()

    # Print sorted functions
    print_sort(bv,sort_func)

# Sort highlighted functions by number of basic blocks
def sort_size(bv):
    # Connect to database
    conn,hash_val = connect_db(bv)
    if conn == None:
        return
    cur = conn.cursor()

    sys.stdout.write('\n')
    sys.stdout.write('DeepReflect: Highlighted functions sorted by number of basic blocks:\n')

    sort_func = list()

    with conn:
        # Get highlighted functions in database
        functions = get_highlight_functions(cur,hash_val)

        for row in functions:
            sample_hash = row[0]
            sample_family = row[1]
            func_addr = row[2]
            func_label = row[3]
            analyst_username = row[5]
            label_type = row[6]

            function = bv.get_function_at(int(func_addr,16))

            # If function exists in this sample
            if function is not None:
                num_bb = len(function.basic_blocks)
            else:
                num_bb = -1

            sort_func.append((sample_hash,sample_family,func_addr,func_label,num_bb,analyst_username,label_type))

    conn.commit()

    # Close connection
    conn.close()

    # Print sorted functions
    print_sort(bv,sort_func)

# Sort highlighted functions by number of callees
def sort_callee(bv):
    # Connect to database
    conn,hash_val = connect_db(bv)
    if conn == None:
        return
    cur = conn.cursor()

    sys.stdout.write('\n')
    sys.stdout.write('DeepReflect: Highlighted functions sorted by number of callees:\n')

    sort_func = list()

    with conn:
        # Get highlighted functions in database
        functions = get_highlight_functions(cur,hash_val)

        for row in functions:
            sample_hash = row[0]
            sample_family = row[1]
            func_addr = row[2]
            func_label = row[3]
            analyst_username = row[5]
            label_type = row[6]

            function = bv.get_function_at(int(func_addr,16))

            count = 0

            # If function exists in this sample
            if function is not None:
                for callee in function.callees:
                    symbol_type = callee.symbol.type

                    if symbol_type in [0,1,2,4,5,6]:
                        count += 1
            else:
                count = -1

            sort_func.append((sample_hash,sample_family,func_addr,func_label,count,analyst_username,label_type))

    conn.commit()

    # Close connection
    conn.close()

    # Print sorted functions
    print_sort(bv,sort_func)

