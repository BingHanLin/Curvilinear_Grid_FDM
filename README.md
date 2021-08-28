# Finite difference method on curvilinear grid

## Introduction
**Curvilinear_Grid_FDM** is project that implement finite difference method on curvilinear grid. The code can be modifide easily by students interested in this topic. More features will be added in the future.

***
## Brief theoretical description



***
## Program Structure

### DiscreteSchemes
There two modules provide necessary informations to assemble discrete schemes.

* **CalCoeff.py**

    Calculate essential coefficients for curvilinear coordinates. A structure mesh is given for computing the coefficients. The available information inlcudes: covariant basis vector, contravariant basis vector, metric tensor and christoffel symbols.

* **OperatorFDM3D.py**

    Genertate the coefficient matrix for 3D finite difference computation. A structure mesh is given for querying the coefficients. The coefficient matrix can be obtained with several kinds of derivative operations.

### Geometry

    Provide the information of a structure mesh.
### Solver
The solvers could be built by the user under specific scenario.


***
## Example
**LAPLACE equation**

Laplace equation is solved in a (1) rectangular, (1) curved rectangular and (3) cylindrical domain with dirichlet boundary conditions on all sides.
![](/asset/rectangular.png)
![](/asset/curvedRectangular.png)
![](/asset/cylindrical.png)


***
## References
[1] [Python unit test](https://docs.python.org/zh-tw/3/library/unittest.html#command-line-interface)

[2] [Python unit test](https://openhome.cc/Gossip/CodeData/PythonTutorial/UnitTestPy3.html)

