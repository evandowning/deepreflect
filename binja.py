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

    # If BNDB file already exists, ignore it
    if os.path.exists(output):
        sys.stderr.write('{0}: Binja file already exists\n'.format(output))
        sys.exit(1)

    # Perform analysis
    start = time.time()
    bv = BinaryViewType.get_view_of_file(fn)
    bv.update_analysis_and_wait()
    sys.stdout.write('{0}: Binja Finished Analysis. Took {1} seconds.\n'.format(fn,time.time() - start))

    # Write out BNDB file
    start = time.time()
    bv.create_database(output, None)
    sys.stdout.write('{0}: Binja Created BNDB file. Took {1} seconds.\n'.format(output,time.time() - start))

if __name__ == '__main__':
    _main()
