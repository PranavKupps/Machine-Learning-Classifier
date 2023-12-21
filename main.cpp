// Project UID db1f506d06d84ab787baf250c265e24e
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <map>
#include <vector>
#include <sstream>
#include <string.h>
#include "csvstream.h"
#include <set>
#include <cmath>
using namespace std;
class Classifier {
private :
    bool debug;
    int total_inputs;
    set<string> all_unique_words;
    map<string, int> posts_with_word;
    string label;
    map<string, int> posts_with_label;
    string return_label;
    pair<string, double> prob;
    map<pair<string, string>, int> plw;
    int numCorrect;
    int totalNum;
    set<string> s;
    int number_of_posts;

    void two_inputs(const string &input) {
        s = unique_words(input);
        for(auto &word: s) {
            all_unique_words.insert(word);
        }
        for(auto &word: all_unique_words) {
            if(s.find(word) != s.end()) {
                posts_with_word[word]++;
            }
        }
    }

    set<string> unique_words(const string &input) {
        istringstream editable_words(input);
        s.clear();
        string word;
        while (editable_words >> word) {
            s.insert(word);
        }
        return s;
    }

    void num_posts_per_label(const string &label) {
        posts_with_label[label]++;
    }

    void posts_label_words(const string &input, const string &label) {
        s = unique_words(input);
        for(auto &word: s) {
            plw[{word, label}]++;
        }
    }

    pair<string, double> probability_per_word(const string &input)  {
        s = unique_words(input);
        double maxProb = -INFINITY;
        for(auto &iter: posts_with_label) {
            double prob = 0;
            string label = iter.first;
            prob += log(1.0*posts_with_label.at(label)/total_inputs);
            for(auto &word: s) {
                if(!posts_with_word.count(word)) {
                    prob += log(1.0/total_inputs);
                }
                else if(!plw.count({word, label})) {
                    prob += log(1.0*posts_with_word.at(word)/total_inputs);
                }
                else {
                    prob += log(1.0*plw.at({word, label})
                                /posts_with_label.at(label));
                }
            }
            if(prob > maxProb) {
                maxProb = prob;
                return_label = label;
            }
        }
        return {return_label, maxProb};
    }
public :
    Classifier(bool debug) :
        debug(debug), total_inputs(0) { }

    void train(string filename) {
        csvstream csvin(filename);
        map<string, string> r;
        if(debug) {
            cout << "training data:" << endl;
        }
        while(csvin >> r) {
            total_inputs++;
            two_inputs(r["content"]);
            num_posts_per_label(r["tag"]);
            posts_label_words(r["content"], r["tag"]);
            if(debug) {
                cout << "  label = " << r["tag"] << ", content = " << r["content"] 
                << endl;
            }
        }
        outputs(3);
    }
    void test(string filename)  {
        cout << "test data:" << endl;
        csvstream csvin(filename);
        numCorrect = 0, totalNum = 0;
        map<string, string> input;
        while(csvin >> input) {
            prob = probability_per_word(input["content"]);
            if(input["tag"] == probability_per_word(input["content"]).first) {
                numCorrect++;
            }
            totalNum++;
            cout << "  correct = " << input["tag"] << ", predicted = " << 
            prob.first << ", log-probability score = " << prob.second << endl;
            cout << "  content = " << input["content"] << endl << endl;
        }
        outputs(1);
    }
    void no_debug(string training_file, string testing_file) {
        train(training_file);
        test(testing_file);
    }
    void debugs() {
        cout << "classes:" << endl;
        for(auto &iter: posts_with_label) {
            label = iter.first;
            number_of_posts = posts_with_label.at(label);
            cout << "  " << label << ", " << number_of_posts << " examples, log-prior = " 
            << log(1.0*number_of_posts/total_inputs) << endl;
        }
        cout << "classifier parameters:" << endl;
        for(auto &iter: posts_with_label) {
            label = iter.first;
            for(auto &word: all_unique_words) {
                if(plw.count({word, label})) {
                    int count = plw.at({word, label});
                    cout << "  " << label << ":" << word << ", count = " << count
                    << ", log-likelihood = " 
                    << log(1.0*count/posts_with_label.at(label)) << endl;
                }
            }
        }
        cout << endl;
    }
    void outputs(int num) {
        if(num == 4) {
            cout << "vocabulary size = " << all_unique_words.size() << endl;
        } else if(num == 6) {
            cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        } else if(num == 3) {
            cout << "trained on " << total_inputs << " examples" << endl;
        } else if(num == 1) {
                    cout << "performance: " << numCorrect << " / " << totalNum << 
        " posts predicted correctly" << endl;
        }
    }
};
int main(int argc, char *argv[]) {
    cout.precision(3);
    ifstream testing_file;
    testing_file.open(argv[2]);
    if(!testing_file.is_open()) {
        cout << "Error opening file: " << argv[2] << endl;
    }
    ifstream training_file;
    training_file.open(argv[1]);
    if(!training_file.is_open()) {
        cout << "Error opening file: " << argv[1] << endl;
    }
    if(argc == 4) {
        if(!strcmp(argv[3], " --debug")) {
            Classifier program(false);
            program.outputs(6);
            program.no_debug(argv[1], argv[2]);
        } else {
            Classifier program(true);
            program.train(argv[1]);
            program.outputs(4);
            cout << endl;
            program.debugs();
            program.test(argv[2]);
        }
    } else if(argc == 3) {
        Classifier program(false);
        program.no_debug(argv[1], argv[2]);
    } else {
        cout << "Usage: main.exe TRAIN_FILE TEST_FILE [--debug]" << endl;
        return 0;
    }
    return 0;
}