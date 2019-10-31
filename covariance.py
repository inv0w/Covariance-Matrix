import numpy as np
import pandas as pd

def mean_calc(data, pos):
    '''Calaculates the mean given a dataset(list of lists) and position variable.
    In this case the pos value is the matrix index x, y or z. (0, 1 or 2)
    '''
    sum = 0
    for i in range(len(data)):
        sum += data[i][pos]
    return sum/len(data)

def covariance_calc(data, pos, pos2):
    '''Calculates the covariance given a dataset(list of lists) and 2 position
    variables.
    '''
    mean_i = mean_calc(data, pos)
    mean_j = mean_calc(data, pos2)
    cov_sum = 0
    cov_ij = 0
    for i in range(len(data)):
        cov_ij = (data[i][pos] - mean_i) * (data[i][pos2] - mean_j)
        cov_sum += cov_ij/(len(data)-1)
    return cov_sum

def matrix_row(data, pos):
    '''Creates a list given a dataset and a position variable. Uses the covariance
    calculation and a matrix position.
    '''
    row = []
    for i in range(len(data[0])):
        row.append(covariance_calc(data, pos, i))
    return row

def cov_matrix(data):
    '''Appends an array(row) to an empty array(matrix grid) and defines the
    size of the matrix from the size of the dataset.
    '''
    matrix = np.empty((0, len(data[0])), int)
    for i in range(len(data[0])):
        row = matrix_row(data, i)
        matrix = np.append(matrix, np.array([row]), axis=0)
    return matrix

def cov_matrix_calculation(data):
    # calculate covariance matrix of the data
    cov_matx = np.cov(data.T)
    return cov_matx

if __name__ == '__main__':
    df = pd.read_csv('Graduate_Admissions.csv')
    df_get = df[['GRE Score', 'TOEFL Score', 'University Rating']]
    dataset = df_get.to_numpy()
    test_data = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 2], [1, 4, 3]])

    print(cov_matrix(dataset))
    print(cov_matrix_calculation(dataset))
