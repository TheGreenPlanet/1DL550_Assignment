///////////////////////////////////////////////////
// Low Level Parallel Programming 2017.
//
// 
//
// The main starting point for the crowd simulation.
//



#undef max
#include "ped_model.h"
#include "MainWindow.h"
#include "ParseScenario.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QApplication>
#include <QTimer>
#include <thread>

#include "PedSimulation.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstring>

#pragma comment(lib, "libpedsim.lib")

#include <stdlib.h>

int main(int argc, char*argv[]) {
	bool timing_mode = 0;
	int i = 1;
	QString scenefile = "scenario.xml";
	Ped::IMPLEMENTATION implementation_to_test = Ped::SEQ;
	Ped::HEATMAP_IMPLEMENTATION heatmap_implementation = Ped::H_SEQ;

	// Argument handling
	while (i < argc)
	{
		if (argv[i][0] == '-' && argv[i][1] == '-')
		{
			if (strcmp(&argv[i][2], "timing-mode") == 0)
			{
				cout << "Timing mode on\n";
				timing_mode = true;
			}
			else if (strcmp(&argv[i][2], "help") == 0)
			{
				cout << "Usage: " << argv[0] << " [--help] [--timing-mode] [scenario]" << endl;
				return 0;
			}
			else if (strcmp(&argv[i][2], "seq") == 0)
			{
				implementation_to_test = Ped::SEQ;
			}
			else if (strcmp(&argv[i][2], "omp") == 0)
			{
				implementation_to_test = Ped::OMP;
			}
			else if (strcmp(&argv[i][2], "pthread") == 0)
			{
				implementation_to_test = Ped::PTHREAD;
			}
			else if (strcmp(&argv[i][2], "vector") == 0)
			{
				implementation_to_test = Ped::VECTOR;
			}
			else if (strcmp(&argv[i][2], "seqmove") == 0)
			{
				implementation_to_test = Ped::SEQMOVE;
			}
			else if (strcmp(&argv[i][2], "move") == 0)
			{
				implementation_to_test = Ped::MOVE;
			}
			else if (strcmp(&argv[i][2], "heatmap-cuda") == 0)
			{
				heatmap_implementation = Ped::H_CUDA;
			}
			// TODO: add all implementations
			else
			{
				cerr << "Unrecognized command: \"" << argv[i] << "\". Ignoring ..." << endl;
			}
		}
		else // Assume it is a path to scenefile
		{
			scenefile = argv[i];
		}

		i += 1;
	}

	switch (implementation_to_test) {
		case Ped::IMPLEMENTATION::SEQ:
			cout << "Testing SEQ implementation" << endl;
			break;
		case Ped::OMP:
			cout << "Testing OMP implementation" << endl;
			break;
		case Ped::PTHREAD:
			cout << "Testing PTHREAD implementation" << endl;
			break;
		case Ped::VECTOR:
			cout << "Testing VECTOR implementation" << endl;
			break;
		case Ped::SEQMOVE:
			cout << "Testing SEQMOVE implementation" << endl;
			break;
		case Ped::MOVE:
			cout << "Testing MOVE implementation" << endl;
			break;
	}

	switch (heatmap_implementation) {
		case Ped::H_SEQ:
			cout << "Testing SEQ heatmap implementation" << endl;
			break;
		case Ped::H_CUDA:
			cout << "Testing CUDA heatmap implementation" << endl;
			break;
	}

	int retval = 0;
	{ // This scope is for the purpose of removing false memory leak positives

		// Reading the scenario file and setting up the crowd simulation model
		Ped::Model model;
		ParseScenario parser(scenefile);
		model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test, heatmap_implementation);

		// Default number of steps to simulate. Feel free to change this.
		const int maxNumberOfStepsToSimulate = 30;
		
				

		// Timing version
		// Run twice, without the gui, to compare the runtimes.
		// Compile with timing-release to enable this automatically.
		if (timing_mode)
		{
            //MainWindow mainwindow(model, timing_mode);
			// Run sequentially

			double fps_seq, fps_target;
			{
				Ped::Model model;
				ParseScenario parser(scenefile);
				model.setup(parser.getAgents(), parser.getWaypoints(), Ped::MOVE, Ped::H_SEQ);
				PedSimulation simulation(model, NULL, timing_mode);
				// Simulation mode to use when profiling (without any GUI)
				std::cout << "Running reference version...\n";
				auto start = std::chrono::steady_clock::now();
				simulation.runSimulation(maxNumberOfStepsToSimulate);
				auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
				fps_seq = ((float)simulation.getTickCount()) / ((float)duration_seq.count())*1000.0;
				cout << "Reference time: " << duration_seq.count() << " milliseconds, " << fps_seq << " Frames Per Second." << std::endl;
			}

			{
				Ped::Model model;
				ParseScenario parser(scenefile);
				model.setup(parser.getAgents(), parser.getWaypoints(), implementation_to_test, heatmap_implementation);
				PedSimulation simulation(model, NULL, timing_mode);
				// Simulation mode to use when profiling (without any GUI)
				std::cout << "Running target version...\n";
				auto start = std::chrono::steady_clock::now();
				simulation.runSimulation(maxNumberOfStepsToSimulate);
				auto duration_target = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
				fps_target = ((float)simulation.getTickCount()) / ((float)duration_target.count())*1000.0;
				cout << "Target time: " << duration_target.count() << " milliseconds, " << fps_target << " Frames Per Second." << std::endl;
			}
			std::cout << "\n\nSpeedup: " << fps_target / fps_seq << std::endl;
			
			

		}
		// Graphics version
		else
		{
            QApplication app(argc, argv);
            MainWindow mainwindow(model);

			PedSimulation simulation(model, &mainwindow, timing_mode);

			cout << "Demo setup complete, running ..." << endl;

			// Simulation mode to use when visualizing
			auto start = std::chrono::steady_clock::now();
			mainwindow.show();
			simulation.runSimulation(maxNumberOfStepsToSimulate);
			retval = app.exec();

			auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - start);
			float fps = ((float)simulation.getTickCount()) / ((float)duration.count())*1000.0;
			cout << "Time: " << duration.count() << " milliseconds, " << fps << " Frames Per Second." << std::endl;
			
		}

		

		
	}

	cout << "Done" << endl;
	cout << "Type Enter to quit.." << endl;
	getchar(); // Wait for any key. Windows convenience...
	return retval;
}
