import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data consists of students scores resulting from the amount of hours they studied.
# x - dependant variable y-independent variable

x = [[0.5], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0]]
y = [[20], [21], [22], [23], [25], [37], [48], [56], [67], [76], [90], [89], [95]]

# using 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
lr = LinearRegression()

#  Polynomial of degree 4
degree = 4

# a method that lets us make a polynomial model:
model = make_pipeline(PolynomialFeatures(degree), lr)
model.fit(X_train, y_train)
expected_y = y_test
predicted_y = model.predict(X_test)

# score to see model accuracy and predicted values
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
print(expected_y)
print(predicted_y)

# plotting figure
plt.title('Graph to show student Test Grade vs. Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Test Grade')
plt.scatter(X_train, y_train, color='red')
plt.plot(x, model.predict(x), color='green')
plt.show()

# After executing coe, a graph will display the polynomial model of degree 4
# along with data from the training set. The graph shows that although there's a
# linear relationship between the the dependant and independent variables, the polynomial model
# can fit curves and is provides more accuracy in predicting the dependant values better.
