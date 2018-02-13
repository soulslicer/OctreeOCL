# OctreeOCL

OctreeOCL is a GPU implementation of nearest neighbour search. We take the original code, break the recursive structure up to allow up to 3x performance. Then the non-recursive version is easily ported to OpenCL. Currently, NN is implemented for GPU. Maybe oneday radius search can be done too. 

We can search randomly 1 million points in 15ms on a Titan Xp.

OpenCL version of:

https://github.com/jbehley/octree

### Deps:

libproj-dev
libpcl-dev
pcl-tools

### Running:

build2/example $PWD/bunny.pcd

### Performance Octree NN search:

```
-1 Million points random search
-PCL Kd Tree - 185ms
-Fast Radius N Search (My implementation) - 124ms 1 core
-Fast Radius N Search (My implementation) - 40ms 8 core
-Fast Radius N Search on 1 Titan Xp - 15ms 1 core (Winner)
-Fast Radius N Search on 1 GTX 1070 - 30ms 1 core (Winner)
```

### Performance Octree Radius search:

```
-20000 points 0.5m search
-PCL Kd Tree - 20419ms
-Fast Radius N Search (Their implementation) - 2006ms 1 core
-Fast Radius N Search (My implementation) - 1667ms 1 core (Winner)
-GPU Version - Not implemented. Someone please implement kthxbai
```
