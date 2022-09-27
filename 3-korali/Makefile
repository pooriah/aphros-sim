m = 128 128 1
bs = 32 32 1
np=8
nmain=3
build = build


run: compile
	#cd $(build) && mpirun -np $(np) ap.mfer
	cd $(build) && mpirun -np $(nmain) main
	
compile: $(build) conf
	cd $(build) && make

$(build): rm
	mkdir -p $(build)

#conf: a.conf base.conf add.conf std.conf
conf:
	cd $(build) && cmake ..
	#cd $(build) && @test -f np || ( c='echo $(np) > np' && echo "$$c" && eval "$$c" )
	#cd $(build) && @test -f tl || ( c='echo $(tl) > tl' && echo "$$c" && eval "$$c" )
	#cd $(build) && @test -f mesh.conf || ( c='ap.part $(m) $(bs) $(np) > mesh.conf' && echo "$$c" && eval "$$c" )
	#cd $(build) && ap.conf2py a.conf > /dev/null

a.conf:
	cd $(build) && ap.create_a_conf
	
add.conf:
	cd $(build) && touch $@

base.conf:
	cd $(build) && ap.create_base_conf

rm:
	rm -rf $(build)

clean::
	rm -vf job.id.last job.status arg job.id
	rm -vf mesh.conf base.conf a.conf par.conf np tl add.conf

cleandat::
	rm -vf *_*.{xmf,h5,raw,vts,vtk,csv} p.pvd
	rm -vf {vx,vy,vz,p,vf,cl,cls,div,omm}_*.dat
	rm -vf trep_*.log
	rm -vf bc.vtk bc_groups.dat eb.vtk
	rm -vf stat.dat stat_summary out out.conf
	rm -vf lsf.o* slurm*.out
	rm -vf core.*

cleanall: clean cleandat
