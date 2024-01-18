
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("/content/cia.csv")

X = data[["CIA1", "CIA2", "CIA3"]]  
y = data["Pass/Fail"]  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = GaussianNB()


model.fit(X_train, y_train)

y_pred = model.predict(X_test)  # You were missing this line for predictions


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


new_data = [[0, 60, 32]] 
prediction = model.predict(new_data)

print("Predicted:", prediction[0])  
if (prediction[0]==0):
  print("Fail")
else:
  print("Pass")  

 
