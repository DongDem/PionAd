# Requirements Library
- numpy
- pandas
- scikit-learn

# Preprocessing data from file GooglePlay_pion.csv
- Check "Rating" column and insert NaN value by median value of "Rating" column
- Remove NaN value from "Last Updated" and "Content Rating" column
- Encode "App", "Genres", "Content Rating" from text value to numeric values
- One-hot coding the "Category" columns to category list
- Remove "$" in "Price" column
- Binary Encoding the "Type" column (Free ->1, Paid -> 0)
- Encode the time values from "Last Updated" column
- Remove "+", "," characters and replace values by 1:4 point to the next level in "Installs" column (multiply 1.8 * current_values)
- Remove "k", "M" in "Size" column and replace value of kbytes to Mbytes. Remove rows with Size == varies_with_devices
- => the modified.csv is preprocessing data.
# Create train and test set
- Create input features contains ['App', 'Rating', 'Reviews', 'Size', 'Type', 'Price', 'Content Rating', 'Genres', 'Last Updated'] and concatenate the one-hot coding the Category column as category list (dimension: 9 +33 = 42)
- Create label contains "Installs" columns
- ==> This problem is regression problem because the output value is continuous numeric values.
- Split train and test data by scikit-learn.train_test_split (8:2 ratio)
- Use scaling techniques to normalize the input data and avoid overfitting problem and increase the testing accuracy.
# Model and Evaluation
- Use the best regressor (Linear Regression, KNN Regressor, GradientBoostingRegressor) model for the dataset
- Error metrics is root mean squared logarithmic error  formula:sum (abs(Log10(Actual) - Log10(Predicted))) => function: rmlse()
