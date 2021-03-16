from binaryninja import *

import sys
from .db import connect_db, get_analyst_id

# Get label id associated with name
def get_label_id(cur,name):
    cur.execute("SELECT id FROM labels WHERE name = %s", (name,))
    label_id = cur.fetchall()

    return label_id

# Get list of labels in database
def get_labels(cur):
    # Fetch this sample's highlighted functions
    cur.execute("SELECT name FROM labels")
    label_rv = cur.fetchall()

    # If no labels have been created yet
    if len(label_rv) == 0:
        labels = list(['unlabeled'])
    else:
        labels = sorted(label_rv)

    return labels

# Get label reviews belonging to this function
def get_reviews(cur,f_id):
    # Fetch this function's reviews
    cur.execute("SELECT l.name FROM functions AS f, reviews AS r, labels AS l WHERE f.id = %s AND r.function_id = f.id AND r.label_id = l.id", (f_id,))
    reviews = cur.fetchall()

    if len(reviews) == 0:
        reviews = list(['unlabeled'])

    return reviews

# Get function by address
def get_function(cur,hash_val,f_addr):
    # Fetch this sample's highlighted functions
    cur.execute("SELECT id,func_addr,cid FROM functions WHERE hash = %s AND func_addr = %s", (hash_val,hex(f_addr)))
    function = cur.fetchall()

    if len(function) == 0:
        show_message_box("File or function not found in DB", "This file has not yet been analyzed or this function has never been highlighted before. Please follow steps to run sample through model & cluster RoI's before using this plugin", MessageBoxButtonSet.OKButtonSet, MessageBoxIcon.ErrorIcon)
        return None

    return function

#TODO
# Removes function label

#TODO
# Modifies function labels
def modify_label(bv, function):
    # Connect to database
    conn,hash_val = connect_db(bv)
    if conn == None:
        return
    cur = conn.cursor()

    # Get all labels in database
    labels = get_labels(cur)
    labels = labels[0]

    # Get this function's address
    func_addr = function.start

    with conn:
        # Get function in database
        row = get_function(cur,hash_val,func_addr)

        if (len(row) == 0) or (row is None):
            return
        # If anything is returned, it will be one row
        row = row[0]

        # Get function address info
        function_id = row[0]
        func_addr = row[1]
        cid = row[2]

        # Get analyst id
        analyst_id = get_analyst_id(cur)
        analyst_id = analyst_id[0]

        # Get current labels for this function
        func_label = get_reviews(cur,function_id)

        try:
            label_selected = ChoiceField("Labels", labels)
            new_label = TextLineField("Enter a new label if function behavior \ndoesn't meet a pre-existing label\n")

            # Determine what label analyst chose
            get_form_input(["Relabel function {0}. Current: {1}".format(func_addr,func_label), None, label_selected, new_label], "Relabel Function")
            label_final = labels[label_selected.result]

            # If analyst manually entered in a new label
            if new_label.result != '':
                label_final = str(new_label.result)


            # If label doesn't exist, add it
            if label_final not in labels:
                cur.execute("INSERT INTO labels (name) VALUES (%s)", (label_final,))

            # Get label id
            label_id = get_label_id(cur,label_final)
            label_id = label_id[0]


            # Add review
            cur.execute("INSERT INTO reviews (function_id, label_id, analyst_id, ts_start) VALUES (%s, %s, %s, %s)", (function_id,label_id,analyst_id, '1970-01-01 00:00'))

            sys.stdout.write('Updating function {0} with label "{1}"\n'.format(func_addr,label_final))

        # If user exits out of plugin window before making a selection
        except TypeError:
            return

    # Close connection
    conn.close()

