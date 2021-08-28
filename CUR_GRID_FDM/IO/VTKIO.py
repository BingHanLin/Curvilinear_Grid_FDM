from CUR_GRID_FDM.Geometry.BaseMesh import BaseMesh

import math
import vtk


class vtkStructuredGridWrapper:
    def __init__(self, mesh: BaseMesh):
        self._mesh = mesh
        self._dims = mesh.mesh_size()

        self._sgrid = vtk.vtkStructuredGrid()
        self._sgrid.SetDimensions(self._dims)

        x = mesh.x_flatten()
        y = mesh.y_flatten()
        z = mesh.z_flatten()

        pt = [0.0] * 3
        points = vtk.vtkPoints()
        points.Allocate(len(mesh.x_flatten()))
        for i in range(len(mesh.x_flatten())):
            pt[0] = x[i]
            pt[1] = y[i]
            pt[2] = z[i]
            points.InsertPoint(i, pt)

        self._sgrid.SetPoints(points)

    def addScalrDataArray(self, name, data):
        if len(data) != self._sgrid.GetNumberOfPoints():
            return

        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(len(data))
        for i in range(len(data)):
            array.SetValue(i, data[i])

        self._sgrid.GetPointData().AddArray(array)

    def write(self, fileName):
        writer = vtk.vtkStructuredGridWriter()
        writer.SetFileName(fileName)
        writer.SetInputData(self._sgrid)
        writer.Write()

    def interact(self):
        sgridMapper = vtk.vtkDataSetMapper()
        sgridMapper.SetInputData(self._sgrid)
        sgridActor = vtk.vtkActor()
        sgridActor.SetMapper(sgridMapper)

        # Create the usual rendering stuff
        renderer = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(renderer)
        renWin.SetWindowName("vtkStructuredGridWrapper interactor")

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        renderer.AddActor(sgridActor)
        renderer.ResetCamera()
        renderer.GetActiveCamera().Elevation(60.0)
        renderer.GetActiveCamera().Azimuth(30.0)
        renderer.GetActiveCamera().Dolly(1.0)
        renWin.SetSize(640, 480)

        # Interact with the data.
        renWin.Render()
        iren.Start()
