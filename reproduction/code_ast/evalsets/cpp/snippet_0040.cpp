//<NAME>
//DSA Assignment 4

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <algorithm>
#include "hash.h"

using namespace std;

// Module function that loads the hashtable passed to it with values from a
// filestream specified within the function. Instructions and error messages are
// written to the console and each line in the file specified is converted to
// lowercase and inserted to the hashtable while the filestream does not return the
// end of file flag.
void loadDictionary(hashTable &dictionaryHash) {
    string dictionaryName;
    ifstream dictionaryFile;
    cout << "Enter name of dictionary: ";
    cin >> dictionaryName;

    dictionaryFile.open(dictionaryName.c_str());
    if (!dictionaryFile) {
        cerr << "Error: could not open " << dictionaryName << endl;
        exit(1);
    }
    while(!dictionaryFile.eof()){
      string line;
      getline(dictionaryFile, line);
      transform(line.begin(), line.end(), line.begin(), ::tolower);
      dictionaryHash.insert(line);
    }
    dictionaryFile.close();
}

// Spellchecks the input file specified in the argument against the hashtable
// passed to it in the argument and prints spellcheck suggestions to the output
// file specified in the argument. While the filestream created by function
// receives characters, it converts the stream to all lowercase letters and checks
// for only valid characters to add to a "word buffer". If any other character
// other than valid characters are received, the word buffer is checked against the
// hashtable using its "contains" member function and then reset. If the word is
// not in the dictionary or is deemed too long, an appropriate suggestion is made
// to the output file stream. As a special case, if the new line character is
// encountered, in addition to the previous steps, the line number value is
// incremented to keep track of the suggestion location. Once the file stream from
// the input returns no characters, the input and output file streams are closed
// and saved.
void spellCheck(hashTable &dictionary, string inputFilename, string outputFilename){
  ifstream inputFile(inputFilename.c_str());
  ofstream outputFile(outputFilename.c_str());
  string wordBuffer = "";
  int lineNumber = 1;
  char analysisChar;

  while(inputFile.get(analysisChar)){
    char rectifiedAnalysisChar = tolower(analysisChar);
    if((rectifiedAnalysisChar == '-') || (rectifiedAnalysisChar == '\'') || ((rectifiedAnalysisChar < 123) && (rectifiedAnalysisChar > 96)) || ((rectifiedAnalysisChar < 58) && (rectifiedAnalysisChar > 47))){
        wordBuffer += rectifiedAnalysisChar;
    }
    else if(rectifiedAnalysisChar == '\n'){
      if(wordBuffer.length() >= 21){
        outputFile << "Long word at line " << lineNumber << ", starts: " << wordBuffer.substr(0,20) << endl;
        wordBuffer = "";
      }
      else{
        if(!dictionary.contains(wordBuffer)){
          outputFile << "Unknown word at line " << lineNumber << ": " << wordBuffer << endl;
        }
      }
      wordBuffer = "";
      lineNumber++;
    }
    else{
      if(wordBuffer.length() >= 21){
        outputFile << "Long word at line " << lineNumber << ", starts: " << wordBuffer.substr(0,20) << endl;
        wordBuffer = "";
      }
      else{
        if(!dictionary.contains(wordBuffer)){
          outputFile << "Unknown word at line " << lineNumber << ": " << wordBuffer << endl;
        }
      }
      wordBuffer = "";
    }
  }
  inputFile.close();
  outputFile.close();
}

// The main function; initializes the proper variables, outputs instructions to the
// console, and times the duration of the spell check operations. The results of
// the timing is displayed to the console while the actual output of the entire
// program is written to a file determined by the user at runtime.
int main(){
  clock_t loadStartTime, loadEndTime, checkStartTime, checkEndTime;
  double loadTime, checkTime;
  string inputFilename, outputFilename;
  hashTable dictionary(50000);

  loadStartTime = clock();
  loadDictionary(dictionary);
  loadEndTime = clock();
  loadTime = ((double) (loadEndTime - loadStartTime)) / CLOCKS_PER_SEC;
  cout << "Total time (in seconds) to load dictionary: " << loadTime << endl;

  cout << "Enter name of input file: ";
  cin >> inputFilename;
  cout << "Enter name of output file: ";
  cin >> outputFilename;

  checkStartTime = clock();
  spellCheck(dictionary, inputFilename, outputFilename);
  checkEndTime = clock();
  checkTime = ((double) (checkEndTime - checkStartTime)) / CLOCKS_PER_SEC;
  cout << "Total time (in seconds) to check document: " << checkTime << endl;

  return 0;
}