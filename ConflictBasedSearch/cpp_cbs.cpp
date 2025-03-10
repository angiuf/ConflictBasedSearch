// ConflictBasedSearch.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <chrono>
#include "LowLevelSolver.h"
#include "HighLevelSolver.h"

std::vector<std::string> split(std::string s, char delim) {
	std::vector<std::string> result;
	std::stringstream ss(s);
	std::string item;

	while (getline(ss, item, delim)) {
		result.push_back(item);
	}
	return result;
}

Map readMap(std::string filename) {

	Map map; // Grid map
	int row, col;

	std::ifstream infile(filename);
	std::string line;

	getline(infile, line); 

	auto splitted = split(line, ' ');
	row = std::stoi(splitted[0]);
	col = std::stoi(splitted[1]);

	getline(infile, line);

	// split second line to get obstacles
	std::istringstream k(line);
	std::string s;

	std::vector<std::vector<Cell>> cells;

	for (int i = 0; i < row; i++) {

		std::vector<Cell> row;
		for (int j = 0; j < col; j++) {
			row.emplace_back(Cell(j, i));
		}
		cells.emplace_back(row);
	};

	while (getline(k, s, ' ')) {

		int obstacleIndex = std::stoi(s);
		int x = obstacleIndex % col;
		int y = obstacleIndex / col;

		cells[y][x].isObstacle = true;
	}

	// next lines are agents. startIndex endIndex
	int start, end;

	std::vector<Agent> agents;

	int agentID = 0;
	while (infile >> start >> end) {

		int startX = start % col;
		int startY = start / col;

		int endX = end % col;
		int endY = end / col;

		Agent agent(agentID);
		agent.start = Cell(startX, startY);
		agent.end = Cell(endX, endY);
		std::cout << "start: "  << startX << startY;
		std::cout << " end: " << endX << endY << "\n";
		agents.emplace_back(agent);
		agentID++;
	}

	map.cells = cells;
	map.agents = agents;
	return map;
}

void printMap(Map map) {
	for (int i = 0; i < map.cells.size(); i++) {
		for (int j = 0; j < map.cells[0].size(); j++) {
			if (map.cells[i][j].isObstacle){
				std::cout << "X";
			}
			else
				std::cout << "_";
		}
		std::cout << std::endl;
	}
}

void printAgents(Map map) {
	for (int i = 0; i < map.agents.size(); i++) {
		std::cout << map.agents[i].start.y;

	}
}

void testLowLevel(Map map) {
	LowLevelSolver solver;
	auto x = solver.findOptimalPaths(std::vector<Constraint>(), map);

	std::cout << x.size();
	for (int i = 0; i < x.size(); i++) {
		for (int j = 0; j < x[0].size(); j++) {
			std::cout << x[i][j].x << "," << x[i][j].y << std::endl;
		}
	}

}

unsigned int Factorial(unsigned int number) {
	return number <= 1 ? number : Factorial(number - 1)*number;
}

void printSolution(std::vector<std::vector<Cell>> optimalPaths) {

	for (auto path : optimalPaths) {
		std::cout << "Optimal path of agent \n";
		for (auto cell : path) {
			std::cout << cell.x << cell.y << "\n";
		}
	}
}


std::vector<std::vector<Cell>> find_path() {
	std::vector<std::vector<Cell>> optimalPaths;
	std::vector<Cell> shoretstPath;
	Map map = readMap("data/map.txt");
	// printMap(map);
	HighLevelSolver solver;
	optimalPaths = solver.solve(map);
	// printSolution(optimalPaths);
	return optimalPaths;
}


// int main() {
//     auto started = std::chrono::high_resolution_clock::now();

// 	std::vector<std::vector<Cell>> optimalPaths;
// 	optimalPaths = find_path();

//     auto done = std::chrono::high_resolution_clock::now();
//     auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();

//     std::cout << elapsedTime << " milliseconds " ;

//     return 0;
// }

