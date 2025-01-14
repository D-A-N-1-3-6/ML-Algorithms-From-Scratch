#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

class NaiveBayes {
private:
    map<string, map<string, int>> wordCounts; // Stores word counts for each class
    map<string, int> classCounts; // Stores the count of each class
    vector<string> classes; // Stores the list of classes
    int totalDocuments; // Stores the total number of documents

public:
    NaiveBayes() {
        totalDocuments = 0;
    }

    void train(const vector<pair<string, string>>& data) {
        for (const auto& doc : data) {
            const string& text = doc.first;
            const string& className = doc.second;

            // Update class counts
            classCounts[className]++;

            // Update word counts for each class
            for (const string& word : split(text)) {
                wordCounts[className][word]++;
            }

            totalDocuments++;
        }

        // Store the list of classes
        for (const auto& entry : classCounts) {
            classes.push_back(entry.first);
        }
    }

    string predict(const string& text) {
        double maxProb = -1.0;
        string predictedClass;

        for (const string& className : classes) {
            double classProb = (double)classCounts[className] / totalDocuments;
            double textProb = calculateTextProbability(text, className);
            double probability = classProb * textProb;

            if (probability > maxProb) {
                maxProb = probability;
                predictedClass = className;
            }
        }

        return predictedClass;
    }

private:
    vector<string> split(const string& str) {
        vector<string> words;
        string word;
        for (char c : str) {
            if (isalnum(c)) {
                word += tolower(c);
            } else if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        }
        if (!word.empty()) {
            words.push_back(word);
        }
        return words;
    }

    double calculateTextProbability(const string& text, const string& className) {
        double textProb = 1.0;
        int classDocCount = classCounts[className];
        for (const string& word : split(text)) {
            int wordCount = wordCounts[className][word];
            textProb *= (wordCount + 1) / (double)(classDocCount + wordCounts[className].size()); // Laplace smoothing
        }
        return textProb;
    }
};

int main() {
    // Example usage
    vector<pair<string, string>> trainingData = {
        {"This is the first document.", "class1"},
        {"This document is the second document.", "class1"},
        {"And this is the third one.", "class2"},
        {"Is this the first document?", "class2"}
    };

    NaiveBayes classifier;
    classifier.train(trainingData);

    string textToPredict = "This is a new document.";
    string predictedClass = classifier.predict(textToPredict);

    cout << "Predicted class: " << predictedClass << endl;

    return 0;
}
