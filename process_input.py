from numpy import *
import csv

# ------------------------------------------------------------------------------------------------
def rescale_data(descriptor_matrix):
    # 1 degree of freedom means (ddof) N-1 unbiased estimation
    descriptors_var = descriptor_matrix.var(axis=0, ddof=1)
    descriptors_mean = descriptor_matrix.mean(axis=0)
    for i in range(0, descriptor_matrix.shape[0]):
        descriptor_matrix[i, :] = (descriptor_matrix[i, :] - descriptors_mean) / sqrt(descriptors_var)
    return descriptor_matrix

# ------------------------------------------------------------------------------------------------
def sort_descriptor_matrix(descriptors, targets):
    # Placing descriptors and targets in ascending order of target (pIC50) value.
    alldata = ndarray((descriptors.shape[0], descriptors.shape[1] + 1))
    alldata[:, 0] = targets
    alldata[:, 1:alldata.shape[1]] = descriptors
    alldata = alldata[alldata[:, 0].argsort()]
    descriptors = alldata[:, 1:alldata.shape[1]]
    targets = alldata[:, 0]

    return descriptors, targets

# ------------------------------------------------------------------------------------------------
# Performs a simple split of the data into training, validation, and testing sets
def simple_split(descriptors, targets):

    testX_indices = [i for i in range(descriptors.shape[0]) if i % 4 == 0]
    validX_indices = [i for i in range(descriptors.shape[0]) if i % 4 == 1]
    trainX_indices = [i for i in range(descriptors.shape[0]) if i % 4 >= 2]

    TrainX = descriptors[trainX_indices, :]
    ValidX = descriptors[validX_indices, :]
    TestX = descriptors[testX_indices, :]

    TrainY = targets[trainX_indices]
    ValidY = targets[validX_indices]
    TestY = targets[testX_indices]

    return TrainX, ValidX, TestX, TrainY, ValidY, TestY
    #return TrainX[:, 0:20], ValidX[:, 0:20], TestX[:, 0:20], TrainY, ValidY, TestY

# ------------------------------------------------------------------------------------------------
def open_descriptor_matrix(fileName):
    preferred_delimiters = [';', '\t', ',', '\n']

    with open(fileName, mode='r') as csvfile:
        # Dynamically determining the delimiter used in the input file
        row = csvfile.readline()

        delimit = ','
        for d in preferred_delimiters:
            if d in row:
                delimit = d
                break

        # Reading in the data from the input file
        csvfile.seek(0)
        datareader = csv.reader(csvfile, delimiter=delimit, quotechar=' ')
        dataArray = array([row for row in datareader if row != ''], order='C')

    if (min(dataArray.shape) == 1):  # flatten arrays of one row or column
        return dataArray.flatten(order='C')
    else:
        return dataArray

# ------------------------------------------------------------------------------------------------
def open_target_values(fileName):
    preferred_delimiters = [';', '\t', ',', '\n']

    with open(fileName, mode='r') as csvfile:
        # Dynamically determining the delimiter used in the input file
        row = csvfile.readline()
        delimit = ','
        for d in preferred_delimiters:
            if d in row:
                delimit = d
                break

        csvfile.seek(0)
        datalist = csvfile.read().split(delimit)
        if ' ' in datalist:
            datalist = datalist[0].split(' ')

    for i in range(datalist.__len__()):
        datalist[i] = datalist[i].replace('\n', '')
        try:
            datalist[i] = float(datalist[i])
        except:
            datalist[i] = datalist[i]

    try:
        datalist.remove('')
    except ValueError:
        no_empty_strings = True

    return datalist

# ------------------------------------------------------------------------------------------------
# Removes constant and near-constant descriptors.
def removeNearConstantColumns(data_matrix, num_unique=10):
    useful_descriptors = [col for col in range(data_matrix.shape[1])
                          if len(set(data_matrix[:, col])) > num_unique]
    filtered_matrix = data_matrix[:, useful_descriptors]

    remaining_desc = zeros(data_matrix.shape[1])
    remaining_desc[useful_descriptors] = 1

    return filtered_matrix, where(remaining_desc == 1)[0]

# ------------------------------------------------------------------------------------------------
# Removes all rows with NaN elements from the data matrix and target matrix.
def removeInvalidRows(descriptors, targets):
    reduced_descriptors = []
    reduced_targets = []
    no_na_values = False

    for r in range(descriptors.__len__()):
        for col in descriptors[r]:
            try:
                float(col)
                no_na_values = True
            except:
                no_na_values = False
                break
        if no_na_values == True:
            reduced_descriptors.append(descriptors[r])
            reduced_targets.append(targets[r])

    return asarray(reduced_descriptors, dtype=float64), \
           asarray(reduced_targets, dtype=float64).flatten(order='C')