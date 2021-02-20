import sys
import os
import argparse
import time

from binaryninja import *

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--exe', help='PE file', required=True, default=False)
    parser.add_argument('--output', help='output file', required=True, default=False)

    args = parser.parse_args()

    # Store arguments
    fn = args.exe
    output = args.output

    # If exe file doesn't exist
    if not os.path.exists(fn):
        sys.stderr.write('{0} does not exist\n'.format(fn))
        sys.exit(1)

    # If bndb file already exists, don't recreate it
    if os.path.exists(output):
        sys.stderr.write('{0}: Binja file already exists\n'.format(output))
        sys.exit(1)

    # Perform analysis
    start = time.time()
    bv = BinaryViewType.get_view_of_file(fn)
    bv.update_analysis_and_wait()
    sys.stdout.write('{0}: Binja finished analysis. Took {1} seconds.\n'.format(fn,time.time() - start))

    # Get folder of bndb file
    # Create it if it doesn't exist
    root = os.path.dirname(output)
    if not os.path.exists(root):
        os.makedirs(root)

    # Write out bndb file
    start = time.time()
    bv.create_database(output, None)
    sys.stdout.write('{0}: Binja created bndb file. Took {1} seconds.\n'.format(output,time.time() - start))

if __name__ == '__main__':
    _main()
