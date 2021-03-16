import sys
from .db import connect_db

# Get highlighted functions
def get_highlight_functions(cur,hash_val):
    functions = list()

    # Fetch this sample's functions' labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, l.name, f.cid, f.score "
                "FROM functions AS f, reviews AS r, labels AS l "
                "WHERE (f.hash = %s) AND (f.cid != -2) AND (f.id = r.function_id) AND (l.id = r.label_id) "
                ,(hash_val,))
    function_labels = cur.fetchall()

    # Fetch this sample's functions' non-labels
    cur.execute("SELECT f.hash, f.family, f.func_addr, 'unlabeled', f.cid, f.score "
                "FROM functions AS f "
                "WHERE (f.hash = %s) AND (f.cid != -2) AND NOT EXISTS (SELECT 1 FROM functions AS f, reviews AS r WHERE f.id = r.function_id) "
                ,(hash_val,))
    function_unlabels = cur.fetchall()

    functions = function_labels + function_unlabels

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
            cid = row[4]
            score = row[5]

            sort_func.append((sample_hash,sample_family,func_addr,func_label,score))

    conn.commit()

    # Close connection
    conn.close()

    sys.stdout.write('{0} | {1}\n'.format('sample hash'.ljust(64),'sample family'.ljust(16)))
    sys.stdout.write('{0} | {1}\n'.format(sample_hash.ljust(64),sample_family.ljust(16)))

    # Sort functions by number of callees
    sys.stdout.write('{0} | {1} | {2}\n'.format('function address'.ljust(18),'function label'.ljust(16),'score'))
    for sample_hash,sample_family,func_addr,func_label,score in sorted(sort_func, key=lambda x: x[-1], reverse=True):
        sys.stdout.write('{0} | {1} | {2}\n'.format(func_addr.ljust(18),func_label.ljust(16),score))
    sys.stdout.write('\n')

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
            cid = row[4]

            sort_func.append((sample_hash,sample_family,func_addr,func_label))

    conn.commit()

    # Close connection
    conn.close()

    # Get function callees
    label = dict()
    count = dict()
    for sample_hash,sample_family,func_addr,func_label in sort_func:
        if func_addr not in count.keys():
            count[func_addr] = 0
            label[func_addr] = func_label

        function = bv.get_function_at(int(func_addr,16))
        num_bb = len(function.basic_blocks)
        count[func_addr] = num_bb

    sys.stdout.write('{0} | {1}\n'.format('sample hash'.ljust(64),'sample family'.ljust(16)))
    sys.stdout.write('{0} | {1}\n'.format(sample_hash.ljust(64),sample_family.ljust(16)))

    # Sort functions by number of callees
    sys.stdout.write('{0} | {1} | {2}\n'.format('function address'.ljust(18),'function label'.ljust(16),'number of basic blocks'))
    for func_addr,_ in sorted(count.items(), key=lambda x: x[1], reverse=True):
        sys.stdout.write('{0} | {1} | {2}\n'.format(func_addr.ljust(18),label[func_addr].ljust(16),count[func_addr]))
    sys.stdout.write('\n')

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
            cid = row[4]

            sort_func.append((sample_hash,sample_family,func_addr,func_label))

    conn.commit()

    # Close connection
    conn.close()

    # Get function callees
    label = dict()
    count = dict()
    for sample_hash,sample_family,func_addr,func_label in sort_func:
        if func_addr not in count.keys():
            count[func_addr] = 0
            label[func_addr] = func_label

        function = bv.get_function_at(int(func_addr,16))

        for callee in function.callees:
            symbol_type = callee.symbol.type

            if symbol_type in [0,1,2,4,5,6]:
                count[func_addr] += 1

    sys.stdout.write('{0} | {1}\n'.format('sample hash'.ljust(64),'sample family'.ljust(16)))
    sys.stdout.write('{0} | {1}\n'.format(sample_hash.ljust(64),sample_family.ljust(16)))

    # Sort functions by number of callees
    sys.stdout.write('{0} | {1} | {2}\n'.format('function address'.ljust(18),'function label'.ljust(16),'number of callees'))
    for func_addr,_ in sorted(count.items(), key=lambda x: x[1], reverse=True):
        sys.stdout.write('{0} | {1} | {2}\n'.format(func_addr.ljust(18),label[func_addr].ljust(16),count[func_addr]))
    sys.stdout.write('\n')

