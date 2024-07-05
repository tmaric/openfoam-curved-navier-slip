import os
import subprocess
import csv

def toExpStr(integer):
    '''Simple exponential latex formatting for plotting.'''
    return "$10^{" + str(integer) + "}$"

def compute_cpp_start_coeffs(bin_name, Sp, Sm, N, tol_cpp, fileName):
    '''Call c++ binary to compute coefficients via shell.'''

    # call c++ utility
    pwd = os.getcwd()
    # call utility as startFlow Sp Sm N tol path/to/store/coeffs.csv
    cmdFlow = bin_name + " " \
                + str(Sp) + " " \
                + str(Sm) + " " \
                + str(N) + " " \
                + str(tol_cpp) + " " \
                + pwd + '''/../../dat/''' + fileName
    # ignoring all command line feedback
    p = subprocess.call(cmdFlow, shell=True)

def compute_errors(bin_path, python_path, cpp_path, rel_path, abs_path):
    '''Call c++ binary to compute relative and absolute errors.'''
    cmdComp = bin_path \
        + " " + python_path + " " + cpp_path + " " + abs_path + " " + rel_path
    subprocess.call(cmdComp, shell=True);

def read_coeffs(filePath):
    '''Read coefficients or errors from filePath.'''
    coeffs = {'Kn':[],'An':[]}
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            coeffs['Kn'].append(float(row[0]))
            coeffs['An'].append(float(row[1]))
    return coeffs
