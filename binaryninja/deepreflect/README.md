# BinaryNinja Plugin
This is the BinaryNinja interface to DeepReflect.

Thank you to [Scott Bergstresser](https://github.com/sab4tg) for creating the alpha version of this plugin.

## Setup
  - Requirements:
    - BinaryNinja 2.3
    - psycopg2
  - Setup:
    ```
    $ cp -r ./binaryninja/deepreflect ~/.binaryninja/plugins/
    ```

## Usage
  - For each binary
    - Display all functions in database
    - Display all functions highlighted by DeepReflect
    - Add/Remove labels for each function
    - Sort highlighted functions by [score, size, number of callees]
  - Summarizing work
    ```
    $ python summary.py
    ```
