from numpy import zeros
from sklearn import linear_model

import fitting_scoring
import process_input

# ------------------------------------------------------------------------------------------------
descriptors_file = "Practice_Descriptors.csv"
targets_file = "Practice_Targets.csv"

# ------------------------------------------------------------------------------------------------
# Step 1
descriptors = process_input.open_descriptor_matrix(descriptors_file)
targets = process_input.open_target_values(targets_file)

# ------------------------------------------------------------------------------------------------
# Step 2
# Filter out molecules with NaN-value descriptors and descriptors with little or no variance
descriptors, targets = process_input.removeInvalidRows(descriptors, targets)
descriptors, active_descriptors = process_input.removeNearConstantColumns(descriptors)
# Rescale the descriptor data
descriptors = process_input.rescale_data(descriptors)

# ------------------------------------------------------------------------------------------------
# Step 3
descriptors, targets = process_input.sort_descriptor_matrix(descriptors, targets)

# ------------------------------------------------------------------------------------------------
# Step 4
X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = process_input.simple_split(descriptors, targets)
data = {'TrainX': X_Train, 'TrainY': Y_Train, 'ValidateX': X_Valid, 'ValidateY': Y_Valid,
        'TestX': X_Test, 'TestY': Y_Test, 'UsedDesc': active_descriptors}

print(str(descriptors.shape[1]) + " valid descriptors and " + str(targets.__len__()) + " molecules available.")

#print(X_Train[0:5, 0:20])

# ------------------------------------------------------------------------------------------------
# Step 5
# Set up the demonstration model
featured_descriptors = [4, 8, 12, 16]  # These indices are "false", applying only to the truncated post-filter descriptor matrix.
binary_model = zeros((1, X_Train.shape[1]))
binary_model[0][featured_descriptors] = 1

#print(binary_model)

# ------------------------------------------------------------------------------------------------
# Step 6
# Create a Multiple Linear Regression object to fit our demonstration model to the data
regressor = linear_model.LinearRegression()
instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'MLR'}

trackDesc, trackFitness, trackModel, \
trackDimen, trackR2train, trackR2valid, \
trackR2test, testRMSE, testMAE, \
testAccPred = fitting_scoring.evaluate_population(model=regressor, instructions=instructions, data=data,
                                                   population=binary_model, exportfile=None)

# ------------------------------------------------------------------------------------------------
# Step 7
for key in trackDesc.keys():
    print("Descriptors:")
    print("\t" + str(trackDesc[key]))  # This will show the "true" indices of the featured descriptors in the full matrix
    print("Fitness:")
    print("\t" + str(trackFitness[key]))
    print("Model:")
    print("\t" + str(trackModel[key]))
    print("Dimensionality:")
    print("\t" + str(trackDimen[key]))
    print("R2_Train:")
    print("\t" + str(trackR2train[key]))
    print("R2_Valid:")
    print("\t" + str(trackR2valid[key]))
    print("R2_Test:")
    print("\t" + str(trackR2test[key]))
    print("Testing RMSE:")
    print("\t" + str(testRMSE[key]))
    print("Testing MAE:")
    print("\t" + str(testMAE[key]))
    print("Acceptable Predictions From Testing Set:")
    print("\t" + str(100*testAccPred[key]) + "% of predictions")

# ------------------------------------------------------------------------------------------------
# Step 8