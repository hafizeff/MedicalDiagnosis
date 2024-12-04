#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

void preprocessData(const std::string &inputFile, const std::string &outputFile) {
    std::ifstream input(inputFile);
    std::ofstream output(outputFile);

    if (!input || !output) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    std::string line;
    std::getline(input, line); // Skip the header row

    // Write header for processed data
    output << "age,gender,fever,cough,headache,fatigue,breathlessness,Diagnosis\n";

    std::unordered_map<std::string, int> genderMap = {{"Male", 0}, {"Female", 1}};
    std::unordered_map<std::string, int> yesNoMap = {{"Yes", 1}, {"No", 0}};

    while (std::getline(input, line)) {
        std::stringstream ss(line);
        std::string age, gender, fever, cough, headache, fatigue, breathlessness, diagnosis;

        // Parse the line
        std::getline(ss, age, ',');
        std::getline(ss, gender, ',');
        std::getline(ss, fever, ',');
        std::getline(ss, cough, ',');
        std::getline(ss, headache, ',');
        std::getline(ss, fatigue, ',');
        std::getline(ss, breathlessness, ',');
        std::getline(ss, diagnosis, ',');

        // Convert categorical data to numerical
        output << age << ","
               << genderMap[gender] << ","
               << yesNoMap[fever] << ","
               << yesNoMap[cough] << ","
               << yesNoMap[headache] << ","
               << yesNoMap[fatigue] << ","
               << yesNoMap[breathlessness] << ","
               << diagnosis << "\n";
    }

    std::cout << "Preprocessing completed and saved to " << outputFile << std::endl;
}

int main() {
    preprocessData("../python/raw_data.csv", "../python/processed_data.csv");
    return 0;
}
