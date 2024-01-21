#ifndef CPP_CBS_H
#define CPP_CBS_H

#include <vector>

Map readMap(const std::string& filename);

// Function declarations
std::vector<std::vector<Cell>> findOptimalPaths(const std::vector<Constraint>& constraints, const Map& map);
unsigned int Factorial(unsigned int number);
void printSolution(const std::vector<std::vector<Cell>>& optimalPaths);
std::vector<std::vector<Cell>> find_path();

#endif // CPP_CBS_H