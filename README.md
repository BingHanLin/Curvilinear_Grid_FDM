# Finite difference method on curvilinear grid

## Introduction
**Curvilinear_Grid_FDM** is project that implement finite difference method on curvilinear grid. The code can be modifide easily by students interested in this topic. More features will be added in the future.

***
## Brief theoretical description
TBD


***
## Program Structure

### CUR_GRID_FDM/DiscreteSchemes
There two modules provide necessary informations to assemble discrete schemes.

* **CalCoeff.py**

    Calculate essential coefficients for curvilinear coordinates. A structure mesh is given for computing the coefficients. The available information inlcudes: covariant basis vector, contravariant basis vector, metric tensor and christoffel symbols.

* **OperatorFDM3D.py**

    Genertate the coefficient matrix for 3D finite difference computation. A structure mesh is given for querying the coefficients. The coefficient matrix can be obtained with several kinds of derivative operations.

### CUR_GRID_FDM/Geometry
Provide the information of a structure mesh.

* **RectangularMesh.py**
    
    Provide a structural rectangular mesh.

* **CurveRectangularMesh.py**
    
    Provide a structural curve rectangular mesh.

* **DonutMesh.py**
    
    Provide a structural donut mesh.


### Solver
The different solvers that need to be built by the user under specific scenario.


***
## Example
**LAPLACE equation**

Laplace equation is solved in a (1) rectangular, (1) curved rectangular and (3) cylindrical domain with dirichlet boundary conditions on all sides.
![](/asset/rectangular.png)
![](/asset/curvedRectangular.png)
![](/asset/cylindrical.png)


## TODO
1. Support user defined mesh.

***
## References
[1] [Finite Difference Coefficients Calculator](https://web.media.mit.edu/~crtaylor/calculator.html)




