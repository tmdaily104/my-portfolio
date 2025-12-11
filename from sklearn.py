from sklearn.metrics import mean_squared_error

# Actual and predicted values
y_true = [649, 253, 370, 148]
y_pred = [623, 253, 150, 237]

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)

print("Mean Squared Error (MSE):", mse)