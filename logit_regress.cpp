#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Sigmoid function: maps values to a range of 0 to 1
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Gradient ascent function for logistic regression
vector<double> gradient_ascent(const vector<double>& X, const vector<double>& y) {
    // Initialize parameters
    double m = 0.0, c = 0.0;  // Slope and y-intercept
    double lr = 0.001;       // Learning rate
    int epochs = 15000;        // Number of iterations

    // Gradient ascent loop
    for (int i = 0; i < epochs; ++i) {
        double dm = 0.0, dc = 0.0;  // Gradients for m and c

        // Calculate gradients for each data point
        for (int j = 0; j < X.size(); ++j) {
            double y_pred = sigmoid(m * X[j] + c);  // Predicted probability
            dc += (y[j] - y_pred);
            dm += (y[j] - y_pred) * X[j];
        }

        // Update parameters using gradients and learning rate
        m += lr * dm / X.size();
        c += lr * dc / X.size();
    }

    return {c, m};  // Return the final values of c and m
}

// Function to predict the probability given X and the model parameters
void predict(double X, const vector<double>& model_params) {
    double y_pred = sigmoid(model_params[0] + X * model_params[1]);
    cout << "Intercept: " << model_params[0] << endl;
    cout << "Slope: " << model_params[1] << endl;
    cout << "Predicted probability: " << y_pred << endl;
}

int main() {
    vector<double> X = {0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50};
    vector<double> y = {0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1};

    // Train the logistic regression model
    vector<double> model_params = gradient_ascent(X, y);

    cout << "Enter a value to predict: ";
    double input_value;
    cin >> input_value;

    // Make a prediction using the trained model
    predict(input_value, model_params);

    return 0;
}
