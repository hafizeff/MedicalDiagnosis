#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <limits>
#include <cstdlib>

// Function to get user input for symptoms as "yes" or "no"
int getYesNoInputAsInt(const std::string& feature) {
    std::string input;
    std::cout << "Does the patient have " << feature << "? (yes/no): ";
    std::cin >> input;

    for (auto &c : input) c = tolower(c);

    if (input == "yes") return 1;
    if (input == "no") return 0;

    std::cerr << "Invalid input! Please enter 'yes' or 'no'." << std::endl;
    return getYesNoInputAsInt(feature);  // Recur if input is invalid
}

// Function to get numeric input for age
double getNumericInput(const std::string& feature) {
    double input;
    std::cout << "Enter the value for " << feature << ": ";
    std::cin >> input;

    if (std::cin.fail()) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cerr << "Invalid input! Please enter a valid numeric value." << std::endl;
        return getNumericInput(feature);
    }

    return input;
}

// Function to get gender input as "male" or "female", convert to 0 or 1
int getGenderInput() {
    std::string input;
    std::cout << "Enter the gender (male/female): ";
    std::cin >> input;

    for (auto &c : input) c = tolower(c);

    if (input == "male") return 0;
    if (input == "female") return 1;

    std::cerr << "Invalid input! Please enter 'male' or 'female'." << std::endl;
    return getGenderInput();  // Recur if input is invalid
}

// Function to write user input to a JSON file for visualization
void writeInputToJson(const std::string& filepath, double age, int gender, int fever, int cough, int headache, int fatigue, int breathlessness) {
    std::ofstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Error opening JSON file for writing input features!" << std::endl;
        return;
    }

    // Write the input features to JSON, excluding name and converting `age` to a numeric value
    file << "[" << age << ", " << gender << ", " << fever << ", " << cough << ", " << headache << ", " << fatigue << ", " << breathlessness << "]" << std::endl;

    file.close();
}

// Function to save input features and diagnosis to a CSV file
void saveToCsv(const std::string& filepath, const std::string& name, double age, int breathlessness, int fatigue, int fever, int headache, int gender, int cough, const std::string& diagnosis) {
    std::ofstream file(filepath, std::ios::app);  // Open in append mode

    if (!file.is_open()) {
        std::cerr << "Error opening CSV file for writing data!" << std::endl;
        return;
    }

    file << name << "," << age << "," << breathlessness << "," << fatigue << "," << fever << "," << headache << "," << gender << "," << cough << "," << diagnosis << "\n";

    file.close();
}

// Function to call the Python visualization script and retrieve the diagnosis
std::string callVisualizationScript() {
    std::cout << "Generating decision path visualization and retrieving diagnosis..." << std::endl;

    // Execute the Python script and redirect output to a temporary file
    int ret = std::system("python3 ../python/visualize_path.py > temp_output.txt");

    // Check if the script executed successfully
    if (ret != 0) {
        std::cerr << "Error executing visualization script." << std::endl;
        return "Unknown Diagnosis"; // Fallback value
    }

    // Open the temporary file to read the diagnosis
    std::ifstream tempFile("temp_output.txt");
    std::string diagnosis;
    if (tempFile.is_open()) {
        std::getline(tempFile, diagnosis); // Read the first line containing the diagnosis
        tempFile.close();
    } else {
        std::cerr << "Error reading output from visualization script." << std::endl;
        return "Unknown Diagnosis"; // Fallback value
    }

    // Remove the temporary file
    std::remove("temp_output.txt");

    
    return diagnosis;
}

// Function to recommend medication based on diagnosis
std::string recommendMedication(const std::string& diagnosis) {
    std::unordered_map<std::string, std::string> medicationMap = {
        {"Flu", "Tamiflu, Relenza, Rapivab."},
        {"Bronchitis", "Albuterol, Tamiflu, Doxycycline."},
        {"Pneumonia", "Macrolides, Zithromax, Fluconazole."},
    };

    return medicationMap[diagnosis];
}

// Function to call the Python prediction script
std::string callPythonPredict(const std::string& inputJsonPath) {
    std::string command = "python3 ../cpp/predict_model.py " + inputJsonPath;
    std::string result;
    char buffer[128];

    // Open a pipe to the Python script
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error calling Python script for prediction!" << std::endl;
        return "Unknown";
    }

    // Read the output
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    // Remove trailing newline characters
    result.erase(result.find_last_not_of("\n\r") + 1);
    return result;
}

// Updated predict function
void predict() {
    std::string name;
    std::cout << "Enter the patient's name: ";
    std::getline(std::cin, name);

    double age = getNumericInput("age");
    int gender = getGenderInput();
    int fever = getYesNoInputAsInt("fever");
    int cough = getYesNoInputAsInt("cough");
    int headache = getYesNoInputAsInt("headache");
    int fatigue = getYesNoInputAsInt("fatigue");
    int breathlessness = getYesNoInputAsInt("breathlessness");

    // Write the input features to JSON
    writeInputToJson("input_features.json", age, gender, fever, cough, headache, fatigue, breathlessness);

    // Call Python script for prediction
    std::string predictedClass =  callVisualizationScript();

    // Output the prediction and recommended medication
    std::cout << "Predicted diagnosis: " << predictedClass << std::endl;
    std::cout << "Recommended medication: " << recommendMedication(predictedClass) << std::endl;

    // Save the input features and diagnosis to CSV
    saveToCsv("patient_data.csv", name, age, breathlessness, fatigue, fever, headache, gender, cough, predictedClass);

    // Call the visualization script
    //callVisualizationScript();
}

int main() {
    // Add header to CSV file if it's empty
    std::ifstream checkFile("patient_data.csv");
    if (!checkFile.is_open() || checkFile.peek() == std::ifstream::traits_type::eof()) {
        std::ofstream file("patient_data.csv");
        file << "Name,Age,Breathlessness,Fatigue,Fever,Headache,Gender,Cough,Diagnosis\n";
        file.close();
    }

    predict();  // Call the predict function
    return 0;
}
