import pandas as pd
import matplotlib.pyplot as plt
import os

STD_CSV_PATH = './vector_api_test/'
STD_PLOT_PATH = './vector_api_test_plots/'
HEADERS = ['n','exec_time_no_simd','exec_time_simd']

def plot_from_csv(sparsity_1=None, sparsity_2=None, k=None):
    # Read CSV file
    file_path = 'performance_' + sparsity_1 + '_' + sparsity_2 + '_k=' + k 
    data = pd.read_csv(STD_CSV_PATH + file_path + '.csv', header=None, names=HEADERS)
    data = data.iloc[1:]
    data = data.reset_index(drop=True)
   
    # Convert strings to int
    data[HEADERS[1]] = data[HEADERS[1]].astype(int)
    data[HEADERS[2]] = data[HEADERS[2]].astype(int)

    # Create the plot directory if it does not exist
    if not os.path.exists(STD_PLOT_PATH):
        os.makedirs(STD_PLOT_PATH)

    # Create and save the plot
    plt.title('Performance test with Sparsity_1 = ' + sparsity_1 + ', Sparsity_2 = ' + sparsity_2 + ', k = ' + k)
    plt.xlabel(HEADERS[0] + '= size of matrix (squared)')
    plt.ylabel('Execution time in ms', rotation=90)

    plt.plot(data[HEADERS[0]], data[HEADERS[1]], color="r", label='Normal Mult')
    plt.plot(data[HEADERS[0]], data[HEADERS[2]], color="b", label='Vector API Mult')

    plt.legend()
    plt.show()
    # plt.savefig(STD_PLOT_PATH + file_path + '.png')

def main():
    sparsity_1 = input('sparsity_1: ')
    sparsity_2 = input('sparsity_2: ')
    k = input('k: ')
    plot_from_csv(sparsity_1, sparsity_2, k)

if __name__ == '__main__':
    main()
