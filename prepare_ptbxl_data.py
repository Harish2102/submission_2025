#!/usr/bin/env python

# Load libraries.
import argparse
import numpy as np
import os
import os.path
import pandas as pd
import shutil
import sys
import wfdb

from helper_code import find_records, get_signal_files, is_integer

# Parse arguments.
def get_parser():
    description = 'Prepare the PTB-XL database for use in the Challenge.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True) # records100 or records500
    parser.add_argument('-d', '--ptbxl_database_file', type=str, required=True) # ptbxl_database.csv
    parser.add_argument('-f', '--signal_format', type=str, required=False, default='dat', choices=['dat', 'mat'])
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Suppress stdout for noisy commands.
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout

def convert_dat_to_mat(record, write_dir=None):
    import wfdb.io.convert
    cwd = os.getcwd()

    # Split into folder and filename
    record_dir = os.path.dirname(record)          # e.g., "00000"
    record_base = os.path.basename(record)        # e.g., "00001_hrm"

    if record_base.endswith('m'):
        return
    # Final folder to write in: write_dir + record_dir
    target_folder = os.path.join(write_dir, record_dir)

    if write_dir:
        os.makedirs(target_folder, exist_ok=True)
        os.chdir(target_folder)

    # Convert the .dat to .mat in the target folder
    with suppress_stdout():
        wfdb.io.convert.matlab.wfdb_to_mat(record_base)

    # Delete original .dat and .hea
    os.remove(record_base + '.hea')
    os.remove(record_base + '.dat')

    # Check converted files exist
    if not os.path.exists(record_base + 'm' + '.hea') or not os.path.exists(record_base + 'm' + '.mat'):
        raise FileNotFoundError(f"Missing: {record_base + 'm'}.mat or .hea in {target_folder}")

    # Rename .hea and .mat to remove 'm'
    os.rename(record_base + 'm' + '.hea', record_base + '.hea')
    os.rename(record_base + 'm' + '.mat', record_base + '.mat')

    # Fix header content
    with open(record_base + '.hea', 'r') as f:
        lines = f.readlines()

    with open(record_base + '.hea', 'w') as f:
        for line in lines:
            if not (line.startswith('#Creator') or line.startswith('#Source')):
                f.write(line.replace(record_base + 'm', record_base))

    # Return to original working directory
    if write_dir:
        os.chdir(cwd)

# Fix the checksums from the Python WFDB library.
def fix_checksums(record, checksums=None):
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)

    header_filename = os.path.join(record + '.hea')
    string = ''
    with open(header_filename, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                arrs = l.split(' ')
                num_leads = int(arrs[1])
            if 0 < i <= num_leads and not l.startswith('#'):
                arrs = l.split(' ')
                arrs[6] = str(checksums[i-1])
                l = ' '.join(arrs)
            string += l

    with open(header_filename, 'w') as f:
        f.write(string)

# Run script.
def run(args):
    # Load the demographic information.
    df = pd.read_csv(args.ptbxl_database_file, index_col='ecg_id')

    # Identify the header files.
    records = [r for r in find_records(args.input_folder) if not r.endswith('m')]

    # Update the header files to include demographics data and copy the signal files unchanged.
    for record in records:

        # Extract the demographics data.
        record_path, record_basename = os.path.split(record)
        ecg_id = int(record_basename.split('_')[0])
        row = df.loc[ecg_id]

        recording_date_string = row['recording_date']
        date_string, time_string = recording_date_string.split(' ')
        yyyy, mm, dd = date_string.split('-')
        date_string = f'{dd}/{mm}/{yyyy}'

        age = row['age']
        age = int(age) if is_integer(age) else float(age)

        sex = row['sex']
        if sex == 0:
            sex = 'Male'
        elif sex == 1:
            sex = 'Female'
        else:
            sex = 'Unknown'

        # Assume that all of the patients are negative for Chagas disease, which is likely to be the case for every or almost every
        # patient in the PTB-XL dataset.
        label = False

        # Specify the label.
        source = 'PTB-XL'

        # Update the header file.
        input_header_file = os.path.join(args.input_folder, record + '.hea')
        output_header_file = os.path.join(args.output_folder, record + '.hea')

        input_path = os.path.join(args.input_folder, record_path)
        output_path = os.path.join(args.output_folder, record_path)

        os.makedirs(output_path, exist_ok=True)

        with open(input_header_file, 'r') as f:
            input_header = f.read()

        lines = input_header.split('\n')
        record_line = ' '.join(lines[0].strip().split(' ')[:4]) + '\n'
        signal_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.strip() and not l.startswith('#')) + '\n'
        comment_lines = '\n'.join(l.strip() for l in lines[1:] \
            if l.startswith('#') and not any((l.startswith(x) for x in ('# Age:', '# Sex:', '# Height:', '# Weight:', '# Chagas label:', '# Source:')))) + '\n'

        record_line = record_line.strip() + f' {time_string} {date_string} ' + '\n'
        signal_lines = signal_lines.strip() + '\n'
        comment_lines = comment_lines.strip() + f'# Age: {age}\n# Sex: {sex}\n# Chagas label: {label}\n# Source: {source}\n'

        output_header = record_line + signal_lines + comment_lines

        with open(output_header_file, 'w') as f:
            f.write(output_header)

        # Copy the signal files if the input and output folders are different.
        if os.path.normpath(args.input_folder) != os.path.normpath(args.output_folder):
            signal_files = get_signal_files(input_header_file)
            for input_signal_file in signal_files:
                output_signal_file = os.path.join(args.output_folder, os.path.relpath(input_signal_file, args.input_folder))
                if os.path.isfile(input_signal_file):
                    shutil.copy2(input_signal_file, output_signal_file)
                else:
                    raise FileNotFoundError(f'{input_signal_file} not found.')

        # Convert data from .dat files to .mat files, if requested.
        if args.signal_format in ('mat', '.mat'):
            convert_dat_to_mat(record, write_dir=args.output_folder)

        # Recompute the checksums as needed.
        fix_checksums(os.path.join(args.output_folder, record))

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))