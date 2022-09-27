#include <stdint.h>
#include <mpi.h>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <atomic>
#include <mutex>

/*#include <aphros/kernel/hydro.h>
#include <aphros/parse/argparse.h>
#include <aphros/parse/parser.h>
#include <aphros/parse/vars.h>
#include <aphros/util/filesystem.h>
#include <aphros/util/module.h>
#include <aphros/distr/distrsolver.h>
#include <aphros/util/git.h>
#include <aphros/util/system.h>*/

#include <unistd.h>

extern const int arc;
extern const char** arv;

//std::mt19937 _randomGenerator;

void env(korali::Sample &s)
{
	std::cout<<"Entering env function\n";
	// Get MPI communicator
	MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
	
	// Get rank and size of subcommunicator
	int rank, size;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&size);
	//std::cout<<rank<<"\t"<<size<<"\n";
	
	// Get rank in world
	int rankGlobal;
	MPI_Comm_rank(MPI_COMM_WORLD,&rankGlobal);
	//std::cout<<rankGlobal<<"\n";
	
	// Setting seed
	size_t sampleId = s["Sample Id"];
	//_randomGenerator.seed(sampleId);

	// Creating results directory
	char resDir[64];
	sprintf(resDir,"%s/sample%03lu",s["Custom Settings"]["Dump Path"].get<std::string>().c_str(),sampleId);
	
	if(rank==0)
	{
		//if(!util::IsDir(resDir))
		{
			mkdir(resDir,0777);
		}
		/*if(!util::IsDir(resDir))
		{
			fprintf(stderr,"[Korali] Error creating results directory for environment: %s.\n", resDir);
			exit(-1);
		}*/
	}
	
	MPI_Barrier(comm);
	
	char path[256];
	getcwd(path,256);
	chdir(resDir);
	
	std::vector<double> action(1,0);
	std::vector<double> state;
	bool done=false;
	double reward;

	state.push_back(0.5);
	
	s["State"]=state;

	int count=0;
	while(!done)
	{
		if(rank==0)
		{
			s.update();
			auto action_s=s["Action"];
			action=action_s.get<std::vector<double>>();
			if(action[0]>0.98)
			{
				action[0]=0.98;
			}
			else if(action[0]<-0.98)
			{
				action[0]=-0.98;
			}
		}
		MPI_Bcast(&action.front(),action.size(),MPI_DOUBLE,0,comm);
		
		state[0]=0.6;
		
		reward=0;
		reward-=abs(state[0]);

		s["State"]=state;
		s["Reward"]=reward;
		//std::cout<<"Count= "<<count++<<"\t"<<s["State"]<<"\t+++++++++\n";
		count++;
		std::cout<<rankGlobal<<"\t"<<rank<<"\t"<<count<<"\n";
		if(count>4)
		{
			std::cout<<"Hello\n";
			done=true;
		}
		
		/*if(rank==0)
		{
			std::cout<<"Reward= "<<reward<<"\n";
			std::cout<<"here \n";
			std::cout<<"Action: "<<action[0]<<"\n";
		}*/
		//MPI_Barrier(comm);
	}
	std::cout<<"Here\n";
	//MPI_Barrier(comm);
	std::cout<<count<<"\n";
	
	//std::cout<<"state= "<<s["State"]<<"\n";
	//if(rank==0)
	{
		s["Termination"]=done?"Terminal":"Truncated";
	}
	std::cout<<"We are out\n";
	MPI_Barrier(comm);
	chdir(path);
}