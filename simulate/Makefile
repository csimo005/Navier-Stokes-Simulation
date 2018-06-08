objects = simulate.o advection.o diffusion.o force.o projection.o

all: $(objects)
	    nvcc -arch=sm_20 $(objects) -o app

%.o: %.cu
	    nvcc -x cu -I. -dc $< -o $@

clean:
	    rm -f *.o app
