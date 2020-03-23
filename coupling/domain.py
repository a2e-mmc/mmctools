import numpy as np
import utm

class Domain(object):
    """Cartesian domain definition with reference lat/lon corresponding
    to the domain origin
    """
    def __init__(self,
            xmin=0.0, ymin=0.0, zmin=0.0,
            xmax=0.0, ymax=0.0, zmax=0.0,
            nx=0,ny=0,nz=0,
            origin_latlon=(45.638004,-120.642973)):
        """
        Parameters
        ----------
        [xyz][min|max] : float
            Domain extents
        nx,ny,nz : int
            Number of cells in each direction
        origin_latlon : list or tuple
            Latitude and longitude of reference location corresponding
            to the grid origin
        """
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.latlon0 = origin_latlon
        self._calc_grid_info()

    def _calc_grid_info(self):
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        self.Lz = self.zmax - self.zmin
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lx / self.nx
        self.x = np.linspace(self.xmin, self.xmax, self.nx+1)
        self.y = np.linspace(self.ymin, self.ymax, self.ny+1)
        self.z = np.linspace(self.zmin, self.zmax, self.nz+1)
        self.xcc = (self.x[1:] + self.x[:-1]) / 2
        self.ycc = (self.y[1:] + self.y[:-1]) / 2
        self.zcc = (self.z[1:] + self.z[:-1]) / 2

    def __repr__(self):
        s = 'Grid extent ({:g} km x {:g} km x {:g} km):\n'.format(
                self.Lx/1000., self.Ly/1000., self.Lz/1000)
        s+= '  x : ({:g}, {:g}), dx={:g}\n'.format(self.xmin,self.xmax,self.dx)
        s+= '  y : ({:g}, {:g}), dy={:g}\n'.format(self.ymin,self.ymax,self.dy)
        s+= '  z : ({:g}, {:g}), dz={:g}\n'.format(self.zmin,self.zmax,self.dz)
        s+= '  mesh size : {:g}M cells'.format(self.nx*self.ny*self.nz/1e6)
        return s

