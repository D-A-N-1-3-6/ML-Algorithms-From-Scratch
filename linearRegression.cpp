#include <iostream>
#include <vector>

using namespace std;

// Function to perform gradient descent
vector<double> gradient_descent(const vector<double>& X, const vector<double>& y) {
    // Initialize parameters
    double m = 0.0, c = 0.0;  // Slope and y-intercept
    double lr = 0.002;       // Learning rate
    int epochs = 1000;        // Number of iterations

    // Gradient descent loop
    for (int i = 0; i < epochs; ++i) {
        double dm = 0.0, dc = 0.0;  // Gradients for m and c

        // Calculate gradients for each data point
        for (int j = 0; j < X.size(); ++j) {
            double y_pred = m * X[j] + c;  // Predicted value
            dc += (y_pred - y[j]);
            dm += (y_pred - y[j]) * X[j];
        }

        // Update parameters using gradients and learning rate
        m -= lr * dm;
        c -= lr * dc;
    }

    return {c, m};  // Return the final values of c and m
}

// Function to predict a value given X and the model parameters
void predict(double X, const vector<double>& model_params) {
    double y_pred = model_params[0] + X * model_params[1];
    cout << "Predicted value: " << y_pred << endl;
}

int main() {
    vector<double> X = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<double> y = {15, 25, 30, 45, 55, 67, 74, 89, 90, 99};

    // Train the model using gradient descent
    vector<double> model_params = gradient_descent(X, y);

    cout << "Enter a value to predict: ";
    double input_value;
    cin >> input_value;

    // Make a prediction using the trained model
    predict(input_value, model_params);

    return 0;
}
