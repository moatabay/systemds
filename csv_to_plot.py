import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

STD_CSV_PATH = './vector_api_test/'
STD_PLOT_PATH = './vector_api_test_plots/'
HEADERS = ['n','k','exec_time_no_simd','exec_time_simd']

def plot_from_csv(file_path, mode=1):
    # Read CSV file
    data = pd.read_csv(STD_CSV_PATH + file_path, header=None, names=HEADERS)
    data = data.iloc[1:]
    data = data.reset_index(drop=True)
   
    # Convert strings to int
    data[HEADERS[1]] = data[HEADERS[1]].astype(int) # k
    data[HEADERS[2]] = data[HEADERS[2]].astype(int) # no_simd
    data[HEADERS[3]] = data[HEADERS[3]].astype(int) # simd

    # Create the plot directory if it does not exist
    if not os.path.exists(STD_PLOT_PATH):
        os.makedirs(STD_PLOT_PATH)

    # Create and save the plot
    plt.title('file = ' + file_path)
    if mode == 1:
        plt.xlabel(HEADERS[0] + '= size of matrix (squared)')
        plt.ylabel('Execution time in ms', rotation=90)
        plt.plot(data[HEADERS[0]], data[HEADERS[2]], color="r", label='Normal Mult')
        plt.plot(data[HEADERS[0]], data[HEADERS[3]], color="b", label='Vector API Mult')
    elif mode == 2:
        plt.xlabel(HEADERS[1] + '= amount of threads')
        plt.ylabel('Execution time in ms', rotation=90)
        plt.plot(data[HEADERS[1]], data[HEADERS[2]], color="r", label='Normal Mult')
        plt.plot(data[HEADERS[1]], data[HEADERS[3]], color="b", label='Vector API Mult')

    plt.legend()
    plt.show()
    # plt.savefig(STD_PLOT_PATH + file_path + '.png')

def main():
    mode = input('Modes: (1) = n on x-axis, (2) = k on x-axis')
    plot_from_csv(sys.argv[1], mode)

if __name__ == '__main__':
    main()
