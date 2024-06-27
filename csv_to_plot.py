import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

STD_PLOT_PATH = './vector_api_test_plots/'
HEADERS = ['rows1','cols1','cols2','k','time_scalar','time_simd','time_mkl','improvement']

def plot_from_csv(file_path, mode=1):
    # Read CSV file
    data = pd.read_csv(file_path, header=None, names=HEADERS)
    data = data.iloc[1:]
    data = data.reset_index(drop=True)

    # Convert strings to int
    data[HEADERS[0]] = data[HEADERS[0]].astype(int) # rows1
    data[HEADERS[1]] = data[HEADERS[1]].astype(int) # cols1
    data[HEADERS[2]] = data[HEADERS[2]].astype(int) # cols2
    data[HEADERS[3]] = data[HEADERS[3]].astype(int) # k
    data[HEADERS[4]] = data[HEADERS[4]].astype(float) # time_scalar
    data[HEADERS[5]] = data[HEADERS[5]].astype(float) # time_simd
    data[HEADERS[6]] = data[HEADERS[6]].astype(float) # time_mkl
    data[HEADERS[7]] = data[HEADERS[7]].astype(float) # improvement

    print(data[HEADERS[0]])
    print(data[HEADERS[1]])
    print(data[HEADERS[2]])
    print(data[HEADERS[3]])
    print(data[HEADERS[4]])
    print(data[HEADERS[5]])
    print(data[HEADERS[6]])
    print(data[HEADERS[7]])

    # Create the plot directory if it does not exist
    if not os.path.exists(STD_PLOT_PATH):
        os.makedirs(STD_PLOT_PATH)

    # Create and save the plot
    plt.title('file = ' + file_path)
    if mode == 1:
        plt.xlabel(HEADERS[0] + '= rows of LHS matrix')
        plt.ylabel('Execution time in ms', rotation=90)
        plt.plot(data[HEADERS[0]], data[HEADERS[4]], color="r", label='Scalar Mult')
        plt.plot(data[HEADERS[0]], data[HEADERS[5]], color="b", label='SIMD Mult')
        plt.plot(data[HEADERS[0]], data[HEADERS[6]], color="g", label='MKL Mult')
    elif mode == 2:
        plt.xlabel(HEADERS[1] + '= columns of LHS matrix')
        plt.ylabel('Execution time in ms', rotation=90)
        plt.plot(data[HEADERS[1]], data[HEADERS[4]], color="r", label='Scalar Mult')
        plt.plot(data[HEADERS[1]], data[HEADERS[5]], color="b", label='SIMD Mult')
        plt.plot(data[HEADERS[1]], data[HEADERS[6]], color="g", label='MKL Mult')
    elif mode == 3:
        plt.xlabel(HEADERS[2] + '= columns of RHS matrix')
        plt.ylabel('Execution time in ms', rotation=90)
        plt.plot(data[HEADERS[2]], data[HEADERS[4]], color="r", label='Scalar Mult')
        plt.plot(data[HEADERS[2]], data[HEADERS[5]], color="b", label='SIMD Mult')
        plt.plot(data[HEADERS[2]], data[HEADERS[6]], color="g", label='MKL Mult')
    elif mode == 4:
        plt.xlabel(HEADERS[3] + '= amount of threads')
        plt.ylabel('Execution time in ms', rotation=90)
        plt.plot(data[HEADERS[3]], data[HEADERS[4]], color="r", label='Scalar Mult')
        plt.plot(data[HEADERS[3]], data[HEADERS[5]], color="b", label='SIMD Mult')
        plt.plot(data[HEADERS[3]], data[HEADERS[6]], color="g", label='MKL Mult')

    plt.legend()
    plt.show()
    # plt.savefig(STD_PLOT_PATH + os.path.basename(file_path) + '.png')

def main():
    mode = int(input('Modes: (1) = rows1 on x-axis, (2) = cols1 on x-axis, (3) = cols2 on x-axis, (4) = k on x-axis: '))
    plot_from_csv(sys.argv[1], mode)

if __name__ == '__main__':
    main()
