import trace1 as G
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay

#method to create rectangular surfaces
def get_surfaces_from_point_cloud(pcd):
    """
        Fits rectangular boxes into a point cloud

        Parameters
        ----------
        pcd_path: String
            path to the point cloud
                  
        Returns
        -------
        surfaces - np.array of G.Surface elements
        """
    #detecting planar patches (boxes with small z dimension)
    planes = pcd.detect_planar_patches(normal_variance_threshold_deg=60, coplanarity_deg=75, outlier_ratio=0.75, min_plane_edge_length=0.0, min_num_points=0, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)) #si potrebbe fare fine tuning
    surfaces = []
    edges = []
    for i in range (8):
        for j in range (i+1,8):
            edges.append([i,j])
    for plane in planes:
        vertices = np.asarray(plane.get_box_points())
        edges = sorted(edges, key= lambda edge: np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
        #identifying the four shortest edges
        midpoints = [np.mean([vertices[edge[0]],vertices[edge[1]]], axis=0) for edge in edges[:4]]
        p0 = midpoints.pop(0)
        midpoints = sorted(midpoints, key= lambda point: np.linalg.norm(point - p0))
        p1 = midpoints[0]
        p2 = midpoints[1]
        points = [p0, p1, p2]
        #the surface is a section of the planar patch
        surfaces.append(G.Surface(np.array(points)))

    return surfaces


#method to optimize surfaces to possibly have also triangular shape or generic parallelogram shape 
#suggestion: to correctly optimize surfaces cut_impurity should be < 1/(steps_per_side-1)
def get_surfaces_from_point_cloud_with_optimization(pcd, cut_impurity = 0.05, steps_per_side = 11):

    """
        Fits rectangular boxes into a point cloud, then tries to reduce the surfaces removing two triangles from each rectangle,
        creating parallelograms or triangles

        triangles are created by taking as vertices two adjacent vertices of the rectangle and a point on the opposite side 

        parallelograms are determined by two opposite vertices of the rectangle and, starting from these vertices, two equal-length
        portions of opposite sides

        if multiple parallelograms/triangles are acceptable, the one with less area is preferred.
        with equal area, the one with less impurity is accepted

        Parameters
        ----------
        pcd_path: String
            path to the point cloud
        cut_impurity: float
            accepted percentage of points in the initial box that are exluded after the reshaping
        steps_per_side: int
            number of triangles and parallelograms attempts for each side
                  

        Returns
        -------
        surfaces - np.array of G.Surface elements
        """

    
    #detecting planar patches (boxes with small z dimension)
    planes = pcd.detect_planar_patches(normal_variance_threshold_deg=60, coplanarity_deg=75, outlier_ratio=0.75, min_plane_edge_length=0.0, min_num_points=0, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)) #si potrebbe fare fine tuning
    surfaces = []
    edges = []
    for i in range (8):
        for j in range (i+1,8):
            edges.append([i,j])
    for plane in planes:
        print(plane)
        vertices = np.asarray(plane.get_box_points())
        
        target = len(plane.get_point_indices_within_bounding_box(pcd.points))
        edges = sorted(edges, key= lambda edge: np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))

        midpoints = [np.mean([vertices[edge[0]],vertices[edge[1]]], axis=0) for edge in edges[:4]]
        p0 = midpoints.pop(0)
        midpoints = sorted(midpoints, key= lambda point: np.linalg.norm(point - p0))
        p1 = midpoints[0]
        p2 = midpoints[1]
        points = [p0, p1, p2]
        #rectangle shape is the baseline
        #best_surface is [parallelogram (boolean), surface defining vertices (np.array (3,3)), area (float)]
        best_surface = [True, points, np.linalg.norm(p1 - p0) * np.linalg.norm(p2 - p0)]


        #separating vertices in face up and face down
        short_edges = edges[:4]

        face_up_vertices = []
        face_down_vertices = []
        for short_edge in short_edges:
            v1 = vertices[short_edge[0]]
            v2 = vertices[short_edge[1]]

            if v1[2] != v2[2]:
                if v1[2] < v2[2]:
                    face_down_vertices.append(v1)
                    face_up_vertices.append(v2)
                else:
                    face_down_vertices.append(v2)
                    face_up_vertices.append(v1)

            else:
                if v1[1] != v2[1]:
                    if v1[1] < v2[1]:
                        face_down_vertices.append(v1)
                        face_up_vertices.append(v2)
                    else:
                        face_down_vertices.append(v2)
                        face_up_vertices.append(v1)

                else:
                    if v1[0] != v2[0]:
                        if v1[0] < v2[0]:
                            face_down_vertices.append(v1)
                            face_up_vertices.append(v2)
                        else:
                            face_down_vertices.append(v2)
                            face_up_vertices.append(v1)

        
        face_up_vertices = sorted(face_up_vertices, key= lambda point: np.linalg.norm(point - face_down_vertices[0]))
        face_down_vertices = sorted(face_down_vertices, key= lambda point: np.linalg.norm(point - face_up_vertices[0]))
        #in both cases 1 is the closest vertex to 0, 2 is the second closest, 3 is the opposite one


        cuts_to_try = np.linspace(0, 1, steps_per_side)

        #trying to cut part of the rectangle to use a triangle
        new_area = np.linalg.norm(face_down_vertices[0] - face_down_vertices[2]) * np.linalg.norm(face_down_vertices[0] - face_down_vertices[1]) * 0.5
        best_impurity = cut_impurity
        if (new_area < best_surface[2]): 
        
            for cut in cuts_to_try:
                
                #noise introduction is necessary to build Delauney mesh (point must not be complanar)
                noise = np.random.normal(scale=1e-10, size=(6, 3))
                hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2], face_down_vertices[1], face_up_vertices[1]])) + noise)
                noise = np.random.normal(scale=1e-10, size=(6, 3))
                hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2], face_down_vertices[2], face_up_vertices[2]]) + noise)
                if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < best_impurity * target):    
                    print("optimized with triangle type 1 with cut = ", cut)#######################
                    best_impurity = sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) / target
                    best_surface = [False, [np.mean([face_down_vertices[1], face_up_vertices[1]], axis=0),  np.mean([ cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2]], axis=0), np.mean([face_down_vertices[3], face_up_vertices[3]], axis=0)], new_area]

                noise = np.random.normal(scale=1e-10, size=(6, 3))
                hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3], face_down_vertices[1], face_up_vertices[1]])) + noise)
                noise = np.random.normal(scale=1e-10, size=(6, 3))
                hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3], face_down_vertices[2], face_up_vertices[2]]) + noise)
                if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < best_impurity * target):    
                    print("optimized with triangle type 2 with cut = ", cut)#######################
                    best_impurity = sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) / target
                    best_surface = [False, [np.mean([face_down_vertices[0], face_up_vertices[0]], axis=0),  np.mean([ cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3]], axis=0), np.mean([face_down_vertices[2], face_up_vertices[2]], axis=0)], new_area]
                        
                noise = np.random.normal(scale=1e-10, size=(6, 3))
                hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0], face_down_vertices[2], face_up_vertices[2]])) + noise)
                noise = np.random.normal(scale=1e-10, size=(6, 3))
                hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0], face_down_vertices[1], face_up_vertices[1]]) + noise)
                if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < best_impurity * target):    
                    print("optimized with triangle type 3 with cut = ", cut)#######################
                    best_impurity = sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) / target
                    best_surface = [False, [np.mean([face_down_vertices[2], face_up_vertices[2]], axis=0),  np.mean([cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0]], axis=0), np.mean([face_down_vertices[3], face_up_vertices[3]], axis=0)], new_area]
                    break

                noise = np.random.normal(scale=1e-10, size=(6, 3))
                hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3], face_down_vertices[2], face_up_vertices[2]])) + noise)
                noise = np.random.normal(scale=1e-10, size=(6, 3))
                hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3], face_down_vertices[1], face_up_vertices[1]]) + noise)
                if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < best_impurity * target):    
                    print("optimized with triangle type 4 with cut = ", cut)#######################
                    best_impurity = sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) / target
                    best_surface = [False, [np.mean([face_down_vertices[0], face_up_vertices[0]], axis=0),  np.mean([cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3]], axis=0), np.mean([face_down_vertices[1], face_up_vertices[1]], axis=0)], new_area]
                    break


        #trying to cut part of the rectangle to use a generic parallelogram
        

        #not trying disadvantageous cuts
        if best_surface[2] == new_area:
            cuts_to_try= cuts_to_try[:int(steps_per_side/2)]

        for cut in cuts_to_try:

            best_impurity = cut_impurity
            new_area = cut *  np.linalg.norm(face_down_vertices[0] - face_down_vertices[2]) * np.linalg.norm(face_down_vertices[0] - face_down_vertices[1])

            parallelogram_optimization_found = False #flag to interrupt cycle if an acceptble parallelogram is found

            noise = np.random.normal(scale=1e-10, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2], face_down_vertices[1], face_up_vertices[1]])) + noise)
            noise = np.random.normal(scale=1e-10, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[3] + (1-cut)*face_down_vertices[1],cut*face_up_vertices[3] + (1-cut)*face_up_vertices[1], face_down_vertices[2], face_up_vertices[2]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < best_impurity * target) and new_area < best_surface[2]:  
                print("optimized with parallelogram type 1 with cut = ", cut)#######################
                parallelogram_optimization_found = True
                best_impurity = sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) / target
                best_surface = [True, [np.mean([face_down_vertices[1], face_up_vertices[1]], axis=0),  np.mean([cut*face_down_vertices[0] + (1-cut)*face_down_vertices[2], cut*face_up_vertices[0] + (1-cut)*face_up_vertices[2]], axis=0), np.mean([cut*face_down_vertices[3] + (1-cut)*face_down_vertices[1],cut*face_up_vertices[3] + (1-cut)*face_up_vertices[1]], axis=0)], new_area]
            
            noise = np.random.normal(scale=1e-10, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[2], face_up_vertices[2], cut*face_down_vertices[2] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[0], face_down_vertices[3], face_up_vertices[3]])) + noise)
            noise = np.random.normal(scale=1e-10, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[1], face_up_vertices[1], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3], face_down_vertices[0], face_up_vertices[0]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < best_impurity * target) and new_area < best_surface[2]:
                print("optimized with parallelogram type 2 with cut = ", cut)#######################
                parallelogram_optimization_found = True
                best_impurity = sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) / target
                best_surface = [True, [np.mean([face_down_vertices[3], face_up_vertices[3]], axis=0),  np.mean([cut*face_down_vertices[2] + (1-cut)*face_down_vertices[0], cut*face_up_vertices[2] + (1-cut)*face_up_vertices[0]], axis=0), np.mean([cut*face_down_vertices[1] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[3]], axis=0)], new_area]            

            noise = np.random.normal(scale=1e-10, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[0], face_up_vertices[0], cut*face_down_vertices[0] + (1-cut)*face_down_vertices[1],cut*face_up_vertices[0] + (1-cut)*face_up_vertices[1], face_down_vertices[2], face_up_vertices[2]])) + noise)
            noise = np.random.normal(scale=1e-10, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[3], face_up_vertices[3], cut*face_down_vertices[3] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[3] + (1-cut)*face_up_vertices[2], face_down_vertices[1], face_up_vertices[1]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < best_impurity * target) and new_area < best_surface[2]:
                print("optimized with parallelogram type 3 with cut = ", cut)#######################
                parallelogram_optimization_found = True
                best_impurity = sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) / target
                best_surface = [True, [np.mean([face_down_vertices[1], face_up_vertices[1]], axis=0),  np.mean([cut*face_down_vertices[0] + (1-cut)*face_down_vertices[1], cut*face_up_vertices[0] + (1-cut)*face_up_vertices[1]], axis=0), np.mean([cut*face_down_vertices[3] + (1-cut)*face_down_vertices[2],cut*face_up_vertices[3] + (1-cut)*face_up_vertices[2]], axis=0)], new_area]

            noise = np.random.normal(scale=1e-10, size=(6, 3))
            hull_1 = Delaunay((np.array([face_down_vertices[1], face_up_vertices[1], cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0],cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0], face_down_vertices[3], face_up_vertices[3]])) + noise)
            noise = np.random.normal(scale=1e-10, size=(6, 3))
            hull_2 = Delaunay(([face_down_vertices[2], face_up_vertices[2], cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3], face_down_vertices[0], face_up_vertices[0]]) + noise)
            if (sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) < best_impurity * target) and new_area < best_surface[2]:
                print("optimized with parallelogram type 4 with cut = ", cut)#######################
                parallelogram_optimization_found = True
                best_impurity = sum(1 for point in pcd.points if (hull_1.find_simplex(point) >= 0 or hull_2.find_simplex(point) >= 0) ) / target
                best_surface = [True, [np.mean([face_down_vertices[3], face_up_vertices[3]], axis=0),  np.mean([cut*face_down_vertices[1] + (1-cut)*face_down_vertices[0], cut*face_up_vertices[1] + (1-cut)*face_up_vertices[0]], axis=0), np.mean([cut*face_down_vertices[2] + (1-cut)*face_down_vertices[3],cut*face_up_vertices[2] + (1-cut)*face_up_vertices[3]], axis=0)], new_area]

            if parallelogram_optimization_found:
                break


        surfaces.append(G.Surface(np.array(best_surface[1]), best_surface[0]))

    return surfaces
