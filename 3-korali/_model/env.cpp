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

#include <aphros/kernel/hydro.h>
#include <aphros/parse/argparse.h>
#include <aphros/parse/parser.h>
#include <aphros/parse/vars.h>
#include <aphros/util/filesystem.h>
#include <aphros/util/module.h>
#include <aphros/distr/distrsolver.h>
#include <aphros/util/git.h>
#include <aphros/util/system.h>

#include <unistd.h>

extern const int arc;
extern const char** arv;

std::mt19937 _randomGenerator;

/*template <class M>
std::vector<double> getstate(DistrSolver<M, Hydro<M>>& ds,int rank)
{*/

template <class M>
void run(MPI_Comm comm,DistrSolver<M, Hydro<M>>& ds)
{
	ds.Run();
	MPI_Barrier(comm);
}

std::vector<double> operator/(std::vector<double> inp,double val)
{
	std::vector<double> out;
	out.resize(inp.size());
	for(int i=0;i<inp.size();i++)
	{
		out[i]=inp[i]/val;
	}
	return(out);
}

template <class M>
std::vector<double> getstate(MPI_Comm comm,DistrSolver<M, Hydro<M>>& ds);

void env(korali::Sample &s)
{
	std::cout<<"Entering env function\n";
	// Get MPI communicator
	MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
	
	// Get rank and size of subcommunicator
	int rank, size;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&size);
	std::cout<<rank<<"\t"<<size<<"\n";
	
	
	// Get rank in world
	int rankGlobal;
	MPI_Comm_rank(MPI_COMM_WORLD,&rankGlobal);
	std::cout<<rankGlobal<<"\n";
	
	
	// Setting seed
	size_t sampleId = s["Sample Id"];
	_randomGenerator.seed(sampleId);

	// Creating results directory
	char resDir[64];
	if(s["Mode"]=="Training")
	{
		sprintf(resDir,"%s/sample%03lu",s["Custom Settings"]["Dump Path"].get<std::string>().c_str(),sampleId);
	}
	else
	{
		sprintf(resDir,"%s/sample%04lu",s["Custom Settings"]["Dump Path"].get<std::string>().c_str(),sampleId);
	}
	if(rank==0)
	{
		if(!util::IsDir(resDir))
		{
			SystemMakeDir(resDir,0);
		}
		if(!util::IsDir(resDir))
		{
			fprintf(stderr,"[Korali] Error creating results directory for environment: %s.\n", resDir);
			exit(-1);
		}
	}
	
	/*FILE* logFile=nullptr;
	if(rank==0)
	{
    	char logFilePath[128];
    	sprintf(logFilePath,"%s/log.txt",resDir);
    	logFile=freopen(logFilePath,"w",stdout);
    	if(logFile==NULL)
    	{
      		printf("[Korali] Error creating log file: %s.\n",logFilePath);
      		exit(-1);
    	}
  	}*/
	
	MPI_Barrier(comm);
	
	char path[256];
	getcwd(path,256);
	//std::cout<<"Current path: "<<path<<"\n";
	chdir(resDir);
	//char temp[256];
	//std::cout<<"We entered: "<<getcwd(temp,256)<<"\n";

	std::vector<double> action(1,0);
	std::vector<double> state;
	std::vector<double> target(1,3.0);
	std::vector<double> history;
	double dt=0.5;
	double tmax=0.7;
	double reward;
	bool done=false;
	
	/*if(rank==0)
	{
		target[0]=4.0;
	}
	MPI_Bcast(&target.front(),target.size(),MPI_DOUBLE,0,comm);
	
	std::cout<<"---- rank= "<<rank<<", "<<target[0]<<"\n";*/
	
	//###########################
	//#####Initialize############
	/*bool isroot=(!rank);
	const auto args=[&arc,&arv,&isroot]()
	{
    	ArgumentParser parser("Distributed solver",isroot);
    	parser.AddVariable<std::string>("config","../../a.conf")
        	.Help("Path to configuration file");
    	return parser.ParseArgs(arc,arv,"--");
  	}();

  	const std::string config = args.String["config"];

  	std::cerr << "Loading config from '" << config << "'" << std::endl;

  	std::map<std::string, std::atomic<int>> var_reads;
  	std::mutex var_reads_mutex;
  	auto hook_read = [&var_reads, &var_reads_mutex](const std::string& key)
  	{
    	auto it = var_reads.find(key);
    	if(it==var_reads.end())
    	{
      		std::lock_guard<std::mutex> guard(var_reads_mutex);
      		it = var_reads.emplace(key, 0).first;
    	}
    	++it->second;
  	};
  	Vars var(hook_read); // parameter storage
  	Parser varparser(var); // parser
  	if(isroot)
	  	std::cout<<"hello\n";
  	varparser.ParseFile(config);
  	//var.Int.Set("verbose", 0);
	FORCE_LINK(init_contang);
	FORCE_LINK(init_vel);
	using M = MeshCartesian<double, 2>;
	typename Hydro<M>::Par par;
	DistrSolver<M, Hydro<M>> ds(comm, var, par);
	var.Double["tmax"]=0.1;
	
	run<M>(comm,ds);
	state=getstate<M>(comm,ds)/var.Double["extent"];
	history.push_back(state[0]);*/
	state.push_back(0.5);
	//#####Initialize End########
	//###########################
	
	s["State"]=state;
	//while(var.Double["tmax"]<tmax && !done)
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
		
		/*var.Vect["gravity"][1]=action[0];
		var.Double["tmax"]+=dt;
		run<M>(comm,ds);
		state=getstate<M>(comm,ds)/var.Double["extent"];
		history.push_back(state[0]);*/
		state[0]=0.6;
		
		reward=0;
		reward-=abs(state[0]-target[0]);
		//reward-=var.Double["tmax"]/tmax;
		
		s["State"]=state;
		s["Reward"]=reward;
		std::cout<<"Count= "<<count++<<"\t"<<s["State"]<<"\t+++++++++\n";
		
		/*if(history.size()>3)
		{
			
			if(abs(history.back()-history[history.size()-2])<0.1 && abs(history[history.size()-3]-history[history.size()-2])<0.1)
			{
				done=false;
				break;
			}
		}*/
		if(count>3)
		{
			break;
		}
		
		
		
		if(rank==0)
		{
			std::cout<<"Reward= "<<reward<<"\n";
			std::cout<<"here \n";
			std::cout<<"Action: "<<action[0]<<"\n";
		}
		MPI_Barrier(comm);
	}
	MPI_Barrier(comm);
	//s.update();
	
	
	std::cout<<"state= "<<s["State"]<<"\n";
	if(rank==0)
	{
		s["Termination"]=done?"Terminal":"Truncated";
	}
	std::cout<<"We are out\n";
	chdir(path);
}

template <class M>
std::vector<double> getstate(MPI_Comm comm,DistrSolver<M, Hydro<M>>& ds)
{
	std::vector<double> state;
	int rank;
	MPI_Comm_rank(comm,&rank);
	
	//using Scal = typename M::Scal;
  	//using EB = Embed<M>;
	void* ptr;
	ds.getkernel(ptr,0);
	std::vector<std::unique_ptr<Hydro<M>>>* hydro;
	hydro=(std::vector<std::unique_ptr<Hydro<M>>>*)ptr;
	
	double vol=0,tvol=0,pos=0,tpos=0;
  	for(long unsigned int i=0;i<hydro->size();i++)
	{	
    	auto& k=*hydro->at(i).get();
    	auto plic=hydro->at(i)->as_->GetPlic();
    	auto& m=hydro->at(i)->m;
    	const auto& ic=m.GetIndexCells();
    	if(auto as=dynamic_cast<typename Hydro<M>::ASV*>(k.as_.get()))
    	{
    		auto& fcvf = as->GetField();
    		for(auto c:m.Cells())
    		{
    			auto id=ic.GetMIdx(c);
    			pos+=fcvf[c]*m.GetVolume(c)*id[1]*k.var.Double["extent"]/k.var.Vect["mesh_size"][0];//x_pos
    			vol+=fcvf[c]*m.GetVolume(c); //volume
    		}
    	}
    }
    
	/*MPI_Reduce(&value,&total_value,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(rank==0)
	{
		std::cout<<"+\t"<<rank<<"\t"<<sqrt(total_value/M_PI)<<"\n";
	}*/
	MPI_Allreduce(&vol,&tvol,1,MPI_DOUBLE,MPI_SUM,comm);
	MPI_Allreduce(&pos,&tpos,1,MPI_DOUBLE,MPI_SUM,comm);
	if(rank==0)
	{
		std::ofstream file;
		file.open("out_pos.txt",std::ios_base::app);
		file<<"volume\t"<<sqrt(tvol/M_PI)<<"\ty_pos\t"<<tpos/tvol<<"\n";
		//std::cout<<"volume\t"<<rank<<"\t"<<sqrt(tvol/M_PI)<<"\n";
		//std::cout<<"y_pos\t"<<rank<<"\t"<<tpos/tvol<<"\n";
	}
	state.push_back(tpos/tvol);
	return(state);
}


/*void env(korali::Sample &s)
{
	std::cout<<"Entering env function\n";
	// Get MPI communicator
	MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
	
	// Get rank and size of subcommunicator
	int rank, size;
	MPI_Comm_rank(comm,&rank);
	MPI_Comm_size(comm,&size);
	std::cout<<rank<<"\t"<<size<<"\n";
	
	
	// Get rank in world
	int rankGlobal;
	MPI_Comm_rank(MPI_COMM_WORLD,&rankGlobal);
	std::cout<<rankGlobal<<"\n";
	
	
	// Setting seed
	size_t sampleId = s["Sample Id"];
	_randomGenerator.seed(sampleId);

	// Creating results directory
	char resDir[64];
	sprintf(resDir,"%s/sample%03u",s["Custom Settings"]["Dump Path"].get<std::string>().c_str(),rankGlobal/size);
	
	if(rank==0)
	{
		if(!util::IsDir(resDir))
		{
			SystemMakeDir(resDir,0);
		}
		if(!util::IsDir(resDir))
		{
			fprintf(stderr,"[Korali] Error creating results directory for environment: %s.\n", resDir);
			exit(-1);
		}
	}
	
	bool done=false;
	std::vector<double> action(1,0);
	std::vector<double> state(1,0);
	
	s["State"]=state;
	if(rank==0)
	{
		s.update();
		auto action_s=s["Action"];
		action=action_s.get<std::vector<double>>();
		std::cout<<"Action: "<<action[0]<<"\n";
	}	
	
}*/

/*void run(MPI_Comm comm, Vars& var);
template <class M>
void run(MPI_Comm comm,DistrSolver<M, Hydro<M>>& ds);


int main(int argc,char** argv)
{
	arv=new const char*[arc];
	for(int i=0;i<arc;i++)
	{
		arv[i]=argv[i];
	}
	
	
	MpiWrapper mpi(&argc, &argv);
  	
  const int rank = mpi.GetCommRank();
  bool isroot = (!rank);
	

  
  const auto args = [&argc, &argv, &isroot]() {
    ArgumentParser parser("Distributed solver", isroot);
    parser.AddVariable<std::string>("config", "a.conf")
        .Help("Path to configuration file");
    return parser.ParseArgs(argc, argv, "--");
  }();

  const std::string config = args.String["config"];

  std::cerr << "Loading config from '" << config << "'" << std::endl;

  std::map<std::string, std::atomic<int>> var_reads;
  std::mutex var_reads_mutex;
  auto hook_read = [&var_reads, &var_reads_mutex](const std::string& key) {
    auto it = var_reads.find(key);
    if (it == var_reads.end()) {
      std::lock_guard<std::mutex> guard(var_reads_mutex);
      it = var_reads.emplace(key, 0).first;
    }
    ++it->second;
  };

  Vars var(hook_read); // parameter storage
  Parser varparser(var); // parser
  std::cout<<"hello\n";
  varparser.ParseFile(config);
  //var.Int.Set("verbose", 0);

  
  
  
  
  
  	//run(mpi.GetComm(),var);
	
	FORCE_LINK(init_contang);
	FORCE_LINK(init_vel);
	using M = MeshCartesian<double, 2>;
	typename Hydro<M>::Par par;
	DistrSolver<M, Hydro<M>> ds(mpi.GetComm(), var, par);
	run<M>(mpi.GetComm(),ds);
	
	var.Double["tmax"]=6.0;
	var.Vect["gravity"][1]=0.98;
	std::cout<<var.Double["tmax"]<<"\n";
	MPI_Barrier(mpi.GetComm());
	
	run<M>(mpi.GetComm(),ds);
	
	return(0);
}

void run(MPI_Comm comm, Vars& var)
{
	//FORCE_LINK(init_contang);
	//FORCE_LINK(init_vel);
	using M = MeshCartesian<double, 2>;
	typename Hydro<M>::Par par;
	
	DistrSolver<M, Hydro<M>> ds(comm, var, par);
	ds.Run();
	//MPI_Barrier(comm);
	
}

template <class M>
void run(MPI_Comm comm,DistrSolver<M, Hydro<M>>& ds)
{
	
	ds.Run();
	MPI_Barrier(comm);
}*/

