import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Gather data on past soccer matches
# For the purposes of this example, let's say we have a CSV file containing the data
data = pd.read_csv('soccer_data.csv')

# Step 2: Clean and pre-process the data
# Remove any rows with missing values
data = data.dropna()

# Convert team names to numerical values
teams = list(set(data['team1']) | set(data['team2']))
team_mapping = {team: i for i, team in enumerate(teams)}
data['team1'] = data['team1'].map(team_mapping)
data['team2'] = data['team2'].map(team_mapping)

# Generate new features
data['goal_difference'] = data['team1_goals'] - data['team2_goals']

# Step 3: Split the data into a training set and a testing set
X = data[['team1', 'team2', 'possession1', 'possession2', 'shots1', 'shots2', 'goal_difference']]
y = data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Choose a machine learning algorithm
model = LogisticRegression()

# Step 5: Train the model using the training data
model.fit(X_train, y_train)

# Step 6: Use the model to make predictions on the testing data
y_pred = model.predict(X_test)

# Step 7: Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Step 8: Use the trained model to make predictions on new, unseen data
new_data = pd.DataFrame([[0, 1, 60, 40, 10, 5, 5], [1, 0, 50, 50, 8, 9, -1]], columns=X.columns)
new_predictions = model.predict(new_data)
print('Predictions:', new_predictions)
