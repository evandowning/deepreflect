from binaryninja import *

import configparser
import hashlib
import psycopg2

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
    config.read('.binaryninja/plugins/deepreflect/db.cfg')

    dbName = config['db']['name']
    dbUser = config['db']['username']
    dbPass = config['db']['password']

    # Get SHA256 value of this file
    hash_val = sha256_checksum(bv)

    conn = None

    # Connect to database
    try:
        conn = psycopg2.connect("dbname='{0}' user='{1}' host='localhost' password='{2}'".format(dbName,dbUser,dbPass))
    except Exception as e:
        sys.stderr.write('{0}\n'.format(str(e)))
        sys.stderr.write('No connection made to db. See log window for details.\n')

    return conn,hash_val

# Get list of labels in database
def get_labels(cur):
    # Fetch this sample's highlighted functions
    cur.execute("SELECT DISTINCT label FROM dr")
    label = cur.fetchall()

    return label

# Get function by address
def get_function(cur,hash_val,f_addr):
    # Fetch this sample's highlighted functions
    cur.execute("SELECT * FROM dr WHERE hash = %s AND func_addr = %s", (hash_val,hex(f_addr)))
    function = cur.fetchall()

    if len(function) == 0:
        show_message_box("File or function not found in DB", "This file has not yet been analyzed or this function has never been highlighted before. Please follow steps to run sample through model & cluster RoI's before using this plugin", MessageBoxButtonSet.OKButtonSet, MessageBoxIcon.ErrorIcon)
        return None

    return function

# Get all functions
def get_all_functions(cur,hash_val):
    # Fetch this sample's highlighted functions
    cur.execute("SELECT * FROM dr WHERE hash = %s", (hash_val,))
    functions = cur.fetchall()

    if len(functions) == 0:
        show_message_box("File not found in DB", "This file has not yet been analyzed. Please follow steps to run sample through model & cluster RoI's before using this plugin", MessageBoxButtonSet.OKButtonSet, MessageBoxIcon.ErrorIcon)
        return None

    return functions

# Get highlighted functions
def get_highlight_functions(cur,hash_val):
    # Fetch this sample's highlighted functions
    cur.execute("SELECT * FROM dr WHERE hash = %s AND cid != -2", (hash_val,))
    functions = cur.fetchall()

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

        sample_hash = functions[0][1]
        sample_family = functions[0][2]
        sys.stdout.write('{0} | {1}\n'.format('sample hash'.ljust(64),'sample family'.ljust(16)))
        sys.stdout.write('{0} | {1}\n'.format(sample_hash.ljust(64),sample_family.ljust(16)))

        sys.stdout.write('{0} | {1}\n'.format('function address'.ljust(18),'function label'.ljust(16)))
        for row in functions:
            func_label = row[3]
            func_addr = row[4]

            sys.stdout.write('{0} | {1}\n'.format(func_addr.ljust(18),func_label.ljust(16)))
        sys.stdout.write('\n')

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

        sample_hash = functions[0][1]
        sample_family = functions[0][2]
        sys.stdout.write('{0} | {1}\n'.format('sample hash'.ljust(64),'sample family'.ljust(16)))
        sys.stdout.write('{0} | {1}\n'.format(sample_hash.ljust(64),sample_family.ljust(16)))

        sys.stdout.write('{0} | {1}\n'.format('function address'.ljust(18),'function label'.ljust(16)))
        for row in functions:
            func_label = row[3]
            func_addr = row[4]

            sys.stdout.write('{0} | {1}\n'.format(func_addr.ljust(18),func_label.ljust(16)))
        sys.stdout.write('\n')

    conn.commit()

    # Close connection
    conn.close()

# Modifies function labels
def modify_label(bv, function):
    # Connect to database
    conn,hash_val = connect_db(bv)
    if conn == None:
        return
    cur = conn.cursor()

    # Get all labels in database
    label_rv = get_labels(cur)
    labels = [l[0] for l in label_rv]

    # Get this function's address
    func_addr = function.start

    with conn:
        # Get function in database
        row = get_function(cur,hash_val,func_addr)

        if len(row) == 0:
            return
        row = row[0]

        # Get function address info
        db_entry_id = row[0]
        func_label = row[3]
        func_addr = row[4]
        cid = row[5]

        try:
            label_selected = ChoiceField("Labels", labels)
            new_label = TextLineField("Enter a new label if function behavior \ndoesn't meet a pre-existing label\n")

            # Determine what label analyst chose
            get_form_input(["Relabel function {0}. Current: {1}".format(func_addr,func_label), None, label_selected, new_label], "Relabel Function")
            label_final = labels[label_selected.result]
            if new_label.result != '':
                label_final = str(new_label.result)


            # Update database with manual label
            cur.execute("UPDATE dr SET label = %s, manually_labeled = True WHERE unique_ID = %s", (label_final, db_entry_id))

            # Update database with cluster shared labels, as long as they haven't been manually labeled yet
            cur.execute("UPDATE dr SET label = %s WHERE manually_labeled = False AND cid = %s", (label_final, cid))

            sys.stdout.write('Updating function {0} with label {1}\n'.format(func_addr,label_final))

        # If user exits out of plugin window before making a selection
        except TypeError:
            return

    # Close connection
    conn.close()

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
            sample_hash = row[1]
            sample_family = row[2]
            func_label = row[3]
            func_addr = row[4]
            cid = row[5]
            score = row[7]

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
            sample_hash = row[1]
            sample_family = row[2]
            func_label = row[3]
            func_addr = row[4]
            cid = row[5]

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
            sample_hash = row[1]
            sample_family = row[2]
            func_label = row[3]
            func_addr = row[4]
            cid = row[5]

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

# Register plugin options
PluginCommand.register("DeepReflect\\1. Show ALL Functions", "Outputs all functions stored in database to BinaryNinja console", display_all)
PluginCommand.register("DeepReflect\\2. Show Highlighted Functions", "Outputs highlighted functions stored in database to BinaryNinja console", display_highlight)
PluginCommand.register_for_function("DeepReflect\\3. Modify Function Label", "Modify function's label", modify_label)
PluginCommand.register("DeepReflect\\4. Sort Function Score", "Sort highlighted functions by score", sort_score)
PluginCommand.register("DeepReflect\\5. Sort Function Size", "Sort highlighted functions by number of basic blocks", sort_size)
PluginCommand.register("DeepReflect\\6. Sort Function Callees", "Sort highlighted functions by number of callees", sort_callee)
