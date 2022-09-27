#include <korali.hpp>

#include "_model2/env.cpp"

const int arc=1;
const char** arv;

int main(int argc,char** argv)
{
	arv=new const char*[arc];
	for(int i=0;i<arc;i++)
	{
		arv[i]=argv[i];
	}
	
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	if (provided != MPI_THREAD_FUNNELED)
	{
		printf("Error initializing MPI\n");
		exit(-1);
	}
	//int nRanks=atoi(argv[argc-1]);
	int nRanks=2,nAgents=1;
	
	// total ranks=k*n+1
	//1 rank for korali, n rank for each worker, and k worker
	int N=1;
	MPI_Comm_size(MPI_COMM_WORLD, &N);
	//N=1;
	N=N - 1; // Minus one for Korali's engine
	N=(int)(N/nRanks);
	
	// Set results path
	std::string trainingResultsPath="_trainingResults/";
	
	// Creating Korali experiment
	auto e = korali::Experiment();
	
	// Check if there is log files to continue training
	auto found=e.loadState(trainingResultsPath+std::string("/latest"));
	if(found==true)
	{
		std::cout<<"[Korali] Continuing execution from previous run...\n";
	}

	// Configuring problem (for test eliminate after)
	e["Problem"]["Type"]="Reinforcement Learning / Continuous";
	e["Problem"]["Environment Function"]=&env;
	e["Problem"]["Agents Per Environment"]=nAgents;

	// Adding custom setting to run the environment without dumping the state files during training
	e["Problem"]["Custom Settings"]["Dump Frequency"]=0.0;
	e["Problem"]["Custom Settings"]["Dump Path"]=trainingResultsPath;
	
	e["Variables"][0]["Name"]=std::string("position");
	e["Variables"][0]["Type"]="State";
	
	e["Variables"][1]["Name"]="gravity";
	e["Variables"][1]["Type"]="Action";
	e["Variables"][1]["Lower Bound"]=-0.98;
	e["Variables"][1]["Upper Bound"]=+0.98;
	e["Variables"][1]["Initial Exploration Noise"]=0.05;
	
	/// Defining Agent Configuration
	e["Solver"]["Type"]="Agent / Continuous / VRACER";
	e["Solver"]["Mode"]="Training";
	e["Solver"]["Episodes Per Generation"]=1;
	//e["Solver"]["Concurrent Environments"]=N;
	e["Solver"]["Experiences Between Policy Updates"]=1;
	e["Solver"]["Learning Rate"]=1e-4;
	e["Solver"]["Discount Factor"]=0.95; // discount factor set to 1
	e["Solver"]["Mini Batch"]["Size"]=128;
	
	/// Defining the configuration of replay memory
	e["Solver"]["Experience Replay"]["Start Size"]=1024;
	e["Solver"]["Experience Replay"]["Maximum Size"]=65536;
	e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
	e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
	e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
	e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

	//// Defining Policy distribution and scaling parameters
	e["Solver"]["Policy"]["Distribution"]="Clipped Normal";
	e["Solver"]["State Rescaling"]["Enabled"] = false;
	e["Solver"]["Reward"]["Rescaling"]["Enabled"] = true;

	// Configuring the neural network and its hidden layers
	e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
	e["Solver"]["Neural Network"]["Optimizer"] = "Adam";

	e["Solver"]["L2 Regularization"]["Enabled"] = true;
	e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

	// recurrent network
	/*e["Solver"]["Time Sequence Length"] = 1; // length of time sequence, corresponding to number of time steps

	e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
	e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

	e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Recurrent/LSTM";
	e["Solver"]["Neural Network"]["Hidden Layers"][1]["Depth"] = 1;
	e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = 128;*/

	// e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
	// e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";
	
	e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  	e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

  	e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  	e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  	e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  	e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128;

  	e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  	e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";
	
	////// Defining Termination Criteria
	e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e7;

	////// Setting Korali output configuration
	e["Console Output"]["Verbosity"] = "Detailed";
	e["File Output"]["Enabled"] = true;
	e["File Output"]["Frequency"] = 1;
	e["File Output"]["Use Multiple Files"] = true;
	e["File Output"]["Path"] = trainingResultsPath;

	////// Running Experiment
	auto k = korali::Engine();

	// Configuring profiler output
	k["Profiling"]["Detail"] = "Full";
	k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
	k["Profiling"]["Frequency"] = 60;

	// set conduit and MPI communicator
	k["Conduit"]["Type"]="Distributed";
	k["Conduit"]["Ranks Per Worker"]=nRanks;
	korali::setKoraliMPIComm(MPI_COMM_WORLD);

	std::cout<<"Before Run"<<std::endl;

	// run korali
	k.run(e);
	
	MPI_Finalize();
	
	
	return(0);
}