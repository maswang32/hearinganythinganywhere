import os
import numpy as np
import plotly.graph_objects as go
import argparse
import rooms.dataset


class Surface:
    """
    Characterizes a planar surface. At the moment, the surface is a triangle or parallelogram.

    Attributes
    ----------
    points: 3x3 np.array
        Each row provides a point on the surface. If the surface is a triangle, these are the points of the triangle.
        If the surface is a parallelogram, the corner must be given first,
        then the diagonal points. The other corner is reflected across the diagonal.
    n: np.array
        normal vector to the plane
    parallelogram: bool
        if the surface is a parallelogram. If it is not, it is assumed to be a triangle.

    Methods
    -------
    project(xyz)
        projects an xyz point onto the plane defined by the surface.
    reflect(xyz)
        reflects an xyz point across the plane defined by the surface.
    """
    def __init__(self, points, parallelogram=True):

        self.points=points

        #If parallelogram, p0 is the corner point, p1, p2 are diagonals
        self.p0 = points[0]
        self.p1 = points[1]
        self.p2 = points[2]

        n = np.cross(self.p1-self.p0 ,self.p2-self.p0)

        self.n = n/np.linalg.norm(n)

        self.parallelogram = parallelogram

        if self.parallelogram:
            self.opposite = self.p1+self.p2-self.p0
        
    def project(self, xyz):
        dist = np.dot(self.n, xyz-self.p0)
        projection = xyz-self.n*dist
        return dist, projection
            
    def reflect(self, xyz):
        dist = np.dot(self.n, xyz-self.p0)
        reflection = xyz-self.n*dist*2
        return reflection


def compute_barycentric(p, a, b, c):
    """
    Given a point p on a triangle defined by 3d points a, b, c, 
    compute the barycentric coordinates u, v, w with respect to a, b, c.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return u, v, w


def line_surface_intersect(line_start, line_end, surface):

    """
    Computes the distance from line_start, along the line defined by [line_start, line_end] to a given plane:

    Parameters
    ----------
    line_start: np.array
        xyz location of the start of the line segment
    line_end: np.array
        xyz location of the end of the line segment
    surface: Surface
        surface we are computing the distance to
    
    Returns
    -------
    d: float
        distance along the line segment to the plane the surface lies in
    intersection point: np.array
        point at which the line intersects the plane the surface lies in
    inside: bool
        if the line intersects plane in the triangle/parallelogram of the surface

    Returns None if the line does not intersect the plane, or if the line given is invalid.
    """
    l = line_end-line_start
    norm = np.linalg.norm(l)
    if norm==0:
        return None
    l = l/norm

    dot = (np.dot(l, surface.n))
    if dot==0:
        return None

    d = np.dot((surface.p0-line_start),surface.n)/dot

    intersection_point = d*l+line_start

    u, v, w = compute_barycentric(intersection_point, surface.p0, surface.p1, surface.p2)

    if surface.parallelogram:
        #v and w correspond to the diagonal points of the parallelogram
        inside = max(u,v,w)<=1 and v>=0 and w>=0
    else:
        inside = max(u,v,w) <= 1 and min(u,v,w) >= 0
         
    return d,intersection_point, inside


def find_reflection_points(surfaces, source, dest, plot=False, save_path=None): #?????metodo non chiaro???????
    """
    Given a set of surfaces, determines if there is a reflection path from the given source to
    the given destination that reflects off each of the surfaces in order.
    If so, gives the points of the reflection path.

    Parameters
    ----------
    surfaces: list of Surfaces
        The list of surfaces we are considering, in order from first to last reflection.
    source: np.array
        xyz of the source location
    dest: np.array
        xyz of the destination location
    plot: bool
        If we want to plot the reflections and the source/dest
    save_path: str or None
        Path to save the plot

    Returns
    -------
    full_distance: float
        Length of the path in meters.
    all_points: np.array
        List of all points the path travels on, of length len(surfaces) + 2.
        Starts with the source point, then all reflection points, then the destination point.
    ds: np.array
        List of line segment distances between the points in all_points.

    Returns None if no path exists between the specified surfaces.
    """
    if plot:
        fig = go.Figure()

    n_surfaces = len(surfaces)

    #Reflect Backwards to get virtual destination
    image = dest
    for i in range(n_surfaces):
        image = surfaces[n_surfaces-i-1].reflect(image)
    
    #Create list of distances, reflection points, and all_points 
    ds = np.zeros((n_surfaces+1))
    reflection_points = np.zeros((n_surfaces, 3))
    all_points = np.zeros((n_surfaces+2, 3))

    #Initialize virtual destination and source
    virtual_dest = image
    virtual_source = source

    if plot:
        plot_surfaces(fig, surfaces)
        plot_points(fig, dest, "Dest")
        plot_points(fig, source, "Source")

    #Compute full path distance
    full_distance = np.linalg.norm(image-source)

    #Iterate forward to find reflection points
    for i in range(n_surfaces):

        #Find where the path intersects the plane
        result = line_surface_intersect(virtual_source, image, surfaces[i])

        #If the line does not intersect, return None
        if result is None:
            return None
        
        #Compute Intersection Point
        d, intersection_point, inside = result

        if plot:
            plot_points(fig, intersection_point, "Reflection")

        #These conditions mean the reflection is invalid
        if not inside or d<0:
            return None

        #Update distances and reflection points
        reflection_points[i] = intersection_point
        ds[i] = d

        #New line segment starts at the computed intersection point
        virtual_source = intersection_point

        #New destination image is reflected across the plane of the current surface
        image = surfaces[i].reflect(image)
    
    #The destination and image should now be the same.
    if np.linalg.norm(image-dest) > 0.01:
        return None
    

    #Last distance is the distance from the final intersection point to the destination
    ds[-1] = np.linalg.norm(dest-intersection_point)
    
    #Add source and dest to path drawn
    all_points[1:-1] = reflection_points
    all_points[0] = source
    all_points[-1] = dest

    if plot:
        plot_lines(fig, all_points)
        if save_path is not None:
            fig.write_image(save_path)
        else:
            fig.show()

    #Sum of segments sould equal full distance
    if abs(np.sum(ds) - full_distance)>0.001:
        return None

    return full_distance, ds, np.array(all_points)


def plot_surfaces(fig, surfaces, color=None, opacity=0.2):
    """
    Plots a list of surface given by surfaces on a figure given by fig.
    Returns the figure.
    """
    for surface in surfaces:
        if surface.parallelogram:
            pts = np.append(surface.points, surface.opposite.reshape(1,-1),axis=0)
            i = np.array([0,3])
            j = np.array([1,1])
            k = np.array([2,2])
        else:
            pts = surface.points
            i = np.array([0])
            j = np.array([1])
            k = np.array([2])

        x, y, z = pts.T

        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, alphahull=5, opacity=opacity, i=i, j=j, k=k, color=color))

        fig.update_scenes(aspectmode='data')
    return fig


def plot_points(fig, xyz,name="",color=None, opacity=1.0,size=None):
    """
    Plots a point or list of points on a figure given by fig.
    Returns the figure.
    """
    if xyz.ndim==1:
        fig.add_trace(go.Scatter3d(x=[xyz[0]], y=[xyz[1]], z=[xyz[2]],
            name=name, mode='markers', marker=dict(color=color, opacity=opacity, size=size)))
    else:
        fig.add_trace(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
            mode='markers', marker=dict(color=color, opacity=opacity, size=size), name=name))
    return fig

def plot_lines(fig, xyzs, color=None): 
    """
    Plots a line that goes through each of the points given by the xyzs
    Returns the figure.
    """   
    xyzs = np.array(xyzs)
    fig.add_trace(go.Scatter3d(x=xyzs[:,0], y=xyzs[:,1], z=xyzs[:,2],mode='lines',line_color=color))
    return fig


def path_trace(path, reflection_surfaces, candidate_transmission_surfaces):
    """
    Given a reflection path, generates a list of points where the path intersects each of the planes, 
    And if the interaction is a transmission or reflection.

    Parameters
    ----------
    path: np.array
        xyz locations that determine a reflection path.
        The first is the source location, the last is the destination. The middle len(reflection_surfaces) are
        the reflection points for each point on reflection_surfaces, in order.
        The format is the same as the all_points array returned by find_reflection_points.
    
    reflection surfaces: list of Surface
        Reflection surfaces in order from first to last.
    
    candidate_transmission_surfaces: list of Surface
        Surfaces to consider for transmission.
    
    Returns
    -------
    points: list of np.array()
        The source point, reflection/transmission points, and the destination point, in order
    interactions: list of string
        Describes the interaction at each point.
    transmission_surface_indices: list of int
        List of the indices in candidate_transmission_surfaces where transmission occurs.
    """
    points = []
    interactions = []
    surfaces = []
    transmission_surface_indices = []
    n_segments = len(path) - 1

    for i in range(n_segments):
        segment_start = path[i]
        segment_end = path[i+1]
        segment_length = np.linalg.norm(segment_start-segment_end)
        points.append(segment_start)

        if i == 0:
            interactions.append("Source")
            surfaces.append(None)
        else:
            interactions.append("Reflection")
            surfaces.append(reflection_surfaces[i-1])

        segment_ds  = []
        segment_transmission_points = []
        segment_transmission_surfaces = []
        segment_transmission_indices = []

        #Go through test surfaces to see which one the segment transmits through
        for idx, candidate_transmission_surface in enumerate(candidate_transmission_surfaces):
            if i < n_segments-1 and candidate_transmission_surface == reflection_surfaces[i]:
                continue

            result = line_surface_intersect(segment_start, segment_end, candidate_transmission_surface)

            if result is None:
                continue

            d, transmission_point, inside = result
            
            if inside and d>0.0001 and d<segment_length-0.0001:
                segment_ds.append(d)
                segment_transmission_points.append(transmission_point)
                segment_transmission_surfaces.append(candidate_transmission_surface)
                segment_transmission_indices.append(idx)

        #If at least one candidate transmission surface is transmitted through
        if len(segment_ds) != 0:
            segment_ds = np.array(segment_ds)
            segment_transmission_indices = np.array(segment_transmission_indices)
            segment_transmission_points = np.array(segment_transmission_points)
            indices = np.argsort(segment_ds)
            segment_transmission_points = list(segment_transmission_points[indices])
            points = points + list(segment_transmission_points)
            surfaces = surfaces + [segment_transmission_surfaces[j] for j in indices]
            transmission_surface_indices = transmission_surface_indices + list(segment_transmission_indices[indices])
            interactions = interactions + ["Transmission"]*len(segment_transmission_points)

        #If this is the last point, append the destination point
        if i==n_segments-1:
            points.append(segment_end)
            interactions.append("Destination")

    return points, interactions, transmission_surface_indices



def gen_axial_indices(index1, index2, order):
    """Generates surface indices of reflection paths between two parallel walls"""
    order_div_2 = order//2
    
    axial_1 = [index1, index2]*order_div_2
    axial_2 = [index2, index1]*order_div_2
    
    if order % 2 == 1:
        axial_1.append(index1)
        axial_2.append(index2)
        
    return [axial_1, axial_2]


def get_reflections_transmissions_and_delays(
    source, dest, surfaces, speed_of_sound=343, max_order=5,
    candidate_transmission_surface_indices=None, img_save_dir=None,
    parallel_surface_pairs = None, max_axial_order=50, return_points=False,
    fs=48000):
    """
    Given surfaces, a source, and destination, traces paths up to a certain order.

    Parameters
    ----------
    source: (3,) array. xyz location of source.
    dest: (3,) array. xyz location of destination.
    surfaces: list of Surface comprising the room's geometry
    speed_of_sound: m/s
    max_order: maximum order to trace reflections up to
    candidate_transmission_surface_indices: surfaces where transmission is possible, surfaces if None
    img_save_dir: if not None, where to save images of the trace
    parallel_surface_pairs: list of list of int, pairs of surface indices.
    max_axial_order: order to trace parallel surface pairs up to
    return_points: if true, just returns points representing the trace
    fs: sampling rate

    Returns
    -------
    reflection_path_indices: list of list
        For each path, the indices of the surfaces it reflects off
    transmission_path_indices: list of list
        For each path, the indices of the surfaces it transmits through
    delays: np.array(N,)
        Delay in samples of each reflection path, proportional to path length
    start_directions: np.array(N,3)
        The first segment of each reflection path.
    end_directions: np.array(N,3)
        Last segment of each reflection path.
    """
    #Generate all possible orderings of reflection surfaces
    all_indices = []
    n_surfaces = len(surfaces)
    if candidate_transmission_surface_indices is None:
        candidate_transmission_surfaces = surfaces
    else:
        candidate_transmission_surfaces = [surfaces[k] for k in candidate_transmission_surface_indices]

    for order in range(1,max_order+1):
        if order == 1:
            for i in range(len(surfaces)):
                all_indices.append([i])
        else:
            for i in range(len(all_indices)):
                for j in range(n_surfaces):
                    pth = all_indices[i]
                    if pth[-1] != j:
                        all_indices.append(pth+[j])
    
    print("Considered Paths:\t" + str(len(all_indices)))


    if parallel_surface_pairs is not None:
        for pair in parallel_surface_pairs:
            for order in range(max_order+1, max_axial_order+1):
                all_indices = all_indices + gen_axial_indices(pair[0], pair[1], order)
    else:
        print("No Axials to Compute")

    #print(all_indices)
    print("Total Considered Paths, after Axial:\t" + str(len(all_indices)))
    reflection_path_indices = []
    transmission_path_indices = []
    distances = []
    start_directions = []
    end_directions = []
    points_for_all_paths = []

    ###############################?????????must be checked if the direct path exists
    
    if direct_path_check(source, dest, surfaces):
        #Direct Path
        reflection_path_indices.append([])
        transmission_path_indices.append([])
        distances.append(np.linalg.norm(source-dest))
        points_for_all_paths.append(np.stack((source,dest)))

        #Direct Path Angle
        direct_ray = dest - source
        start_directions.append(direct_ray)
        end_directions.append(direct_ray)

    

    ###########################################

    #Checks each candidate reflection path for validity, and finds transmission surfaces
    count=0

    if img_save_dir is not None and not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    
    for test_path in all_indices:
        test_surfaces = [surfaces[k] for k in test_path]
        result = find_reflection_points(test_surfaces, source, dest)

        count=count+1
        if result is not None:
            full_distance, _, path = result
            points, interactions, transmission_surface_indices = path_trace(path,
                                                                            test_surfaces,
                                                                            candidate_transmission_surfaces)

            if return_points and len(transmission_surface_indices)==0:
                points_for_all_paths.append(points)


            distances.append(full_distance)
            reflection_path_indices.append(test_path)
            
            start_direction = points[1] - points[0]
            start_directions.append(start_direction)

            end_directions.append(points[-1] - points[-2])

            if candidate_transmission_surface_indices is not None:
                transmission_surface_indices = [
                    surfaces[candidate_transmission_surface_indices[index]] for index in transmission_surface_indices
                    ]

            transmission_path_indices.append(transmission_surface_indices)

            if img_save_dir is not None:
                save_path = img_save_dir+str(count)+".png"
                fig = go.Figure()
                plot_surfaces(fig, test_surfaces, color='cyan')
                actual_transmission_surfaces = [
                    candidate_transmission_surfaces[j] for j in transmission_surface_indices
                    ]
                plot_surfaces(fig, actual_transmission_surfaces, color='red')
                interactions = np.array(interactions)
                plot_points(fig, np.array(points)[np.where(interactions=="Reflection")],
                            color='cyan', name="Reflection Points")
                plot_points(fig, np.array(points)[np.where(interactions=="Transmission")],
                            color='red', opacity=0.7, name="Transmission Points")
                plot_points(fig, source, color='green', name="Source")
                plot_points(fig, dest, color='orange', name="Dest")
                plot_lines(fig, np.array(points))
                plot_lines(fig, np.array([points[-2], end_directions[-1] + points[-2]]), color='black')
                plot_points(fig, np.array([end_directions[-1] + points[-2]]), color='black', size=1)
                plot_lines(fig, np.array( [points[0], points[0]+start_direction]), color = 'hotpink')
                plot_points(fig, np.array([points[0]+start_direction]), color='hotpink',size=1)

                fig.update_layout(
                    title="Distance " + str(full_distance)
                )
                                
                save_path = os.path.join(img_save_dir, str(count)+".png")
                fig.write_image(save_path)

    print("Valid Paths:\t" + str(len(distances)))
    delays = (fs*(np.array(distances)/speed_of_sound)).astype(int)

    if not return_points:
        return (reflection_path_indices, transmission_path_indices,
        np.array(delays), np.array(start_directions), np.array(end_directions))
    else:
        return points_for_all_paths


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', help='Folder to save results in')
    parser.add_argument('dataset', help='Name of Dataset in the Dataloader Class')
    parser.add_argument('--max_order', type=int, default = 5, help='Maximum Reflection Order')
    parser.add_argument('--max_axial_order', type=int, default = 50, help='Max Reflection Order for Parallel Surfaces')
    parser.add_argument('--test', action='store_true', default=False, help='Flat Axial Reflection Order')
    parser.add_argument('--start_from',type=int, default=0,
                        help='Datapoint index to start tracing from - used when computation was partially done')
    parser.add_argument('--train_only',action='store_true', default=False, help='Only trace the training set')

    args = parser.parse_args()

    D = rooms.dataset.dataLoader(args.dataset)

    if not args.test:
        for subdir in ["reflections", "transmissions", "delays", "starts", "ends"]:
            directory = os.path.join(args.save_dir,subdir)
            if not os.path.exists(directory):
                os.makedirs(directory)

        if not args.train_only:
            indices_to_compute = range(args.start_from, D.xyzs.shape[0])
        else:
            indices_to_compute = D.train_indices

        for i in indices_to_compute:
            
            if i<args.start_from:
                continue

            print(i, flush=True)
            reflections, transmissions, delays, start_directions, end_directions = (
                get_reflections_transmissions_and_delays(source=D.speaker_xyz,
                                                         dest=D.xyzs[i],
                                                         surfaces=D.all_surfaces,
                                                         speed_of_sound=D.speed_of_sound,
                                                         max_order=args.max_order,
                                                         parallel_surface_pairs = D.parallel_surface_pairs,
                                                         max_axial_order = args.max_axial_order)
            )

            np.save(os.path.join(args.save_dir,"reflections/"+str(i)+".npy"), np.array(reflections, dtype=object))
            np.save(os.path.join(args.save_dir,"transmissions/"+str(i)+".npy"), np.array(transmissions, dtype=object))
            np.save(os.path.join(args.save_dir,"delays/"+str(i)+".npy"), delays)
            np.save(os.path.join(args.save_dir,"starts/"+str(i)+".npy"), start_directions)
            np.save(os.path.join(args.save_dir,"ends/"+str(i)+".npy"), end_directions)
    else:
        # If test, select an arbitrary point and plot reflection paths.
        print(D.xyzs.shape)
        _ = get_reflections_transmissions_and_delays(
            source=D.speaker_xyz,
            dest=D.xyzs[70],
            surfaces=D.all_surfaces,
            speed_of_sound=D.speed_of_sound,
            max_order=args.max_order,
            img_save_dir=args.save_dir,
            parallel_surface_pairs = D.parallel_surface_pairs,
            max_axial_order = args.max_axial_order)



#############################################################################
def direct_path_check(source, dest, surfaces):
    for surface in surfaces:
        if surface.parallelogram:
            if moller_trumbore_intersect(source, dest, surface.p0, surface.p1, surface.p2):
                return False
            if moller_trumbore_intersect(source, dest, surface.opposite, surface.p1, surface.p2):
                return False
        
        else:
            if moller_trumbore_intersect(source, dest, surface.p0, surface.p1, surface.p2):
                return False
            
    return True


#################################################################################

#############################################################################
def moller_trumbore_intersect(a, b, v0, v1, v2):
    # Definisci gli edge del triangolo
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(b - a, edge2)
    det = np.dot(edge1, h)

    # Controlla se il segmento è parallelo al triangolo
    if abs(det) < 1e-7:
        return False

    inv_det = 1.0 / det
    s = a - v0
    u = inv_det * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = inv_det * np.dot(b - a, q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = inv_det * np.dot(edge2, q)

    if t >= 0.0 and t <= 1.0:
        # Il segmento interseca il triangolo
        return True

    return False
##########################################################################