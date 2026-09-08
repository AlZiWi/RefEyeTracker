import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage
import sklearn
import scipy
from matplotlib import patches
from scipy.spatial.transform import Rotation as R
import requests
import ipympl
from skspatial.objects import Line, Sphere
import sympy
import itertools
np.set_printoptions(legacy='1.25')

# Utils

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def perspective_project(p, focal_length_mm):
    x_proj = p[0] * (focal_length_mm / (p[2]))
    y_proj = p[1] * (focal_length_mm / (p[2]))
    
    return np.array([x_proj, y_proj])
    

def perspective_unproject(p, z, focal_length_mm):
    x_unproj = p[0] * (z / focal_length_mm)
    y_unproj = p[1] * (z / focal_length_mm)
    
    return np.array([x_unproj, y_unproj, z])

def calculate_surface_normal(ellipse, focal_length_mm, r=None, debug=False):
    # Focal length and ellipse parameters in mm -> convert from pixels beforehand
    # Ellipse parameters are expected as diameter, not radius
    # TODO Combination of one axis centered at origin and ellipse angle = 0 leads to divide by zero in T1 / limini calculation
    # This likely implies that parts of this method are not numerically stable for some ellipse parameters
    # TODO pupil center calculation seems to be off for some ellipse parameters -> check derivation again
    
    # calculate ellipse coefficients from axes and angle
    ex = ellipse[0][0]
    ey = ellipse[0][1]
    ea = ellipse[1][0] / 2  # convert to radius
    eb = ellipse[1][1] / 2  # convert to radius
    theta = np.deg2rad(ellipse[2])
    
    if debug:
        print(f"Ellipse params: ex: {ex}, ey: {ey}, ea: {ea}, eb: {eb}, theta (rad): {theta}")

    # (Wiki Ellipse)
    Ad = ((ea**2) * (np.sin(theta)**2)) + ((eb**2) * (np.cos(theta)**2))
    Bd = 2 * ((eb**2) - (ea**2)) * np.sin(theta) * np.cos(theta)
    Cd = ((ea**2) * (np.cos(theta)**2)) + ((eb**2) * (np.sin(theta)**2))
    Dd = - (2 * Ad * ex) - (Bd * ey)
    Ed = - (Bd * ex) - (2 * Cd * ey)
    Fd = (Ad * (ex**2)) + (Bd * ex * ey) + (Cd * (ey**2)) - ((ea**2) * (eb**2))
    
    if debug:
        print(f"Ellipse coefficients: Ad: {Ad}, Bd: {Bd}, Cd: {Cd}, Dd: {Dd}, Ed: {Ed}, Fd: {Fd}")
    
    ad = Ad
    hd = Bd / 2
    bd = Cd
    gd = Dd / 2
    fd = Ed / 2
    dd = Fd

    # camera coordinate system
    gamma = -focal_length_mm  # negative effective focal length -e
    alpha = 0
    beta = 0

    # (3)
    a = gamma**2 * ad
    b = gamma**2 * bd
    c = (ad * alpha**2) + (2 * hd * alpha * beta) + (bd * beta**2) + (2 * gd * alpha) - (2 * fd * beta) + dd
    d = gamma**2 * dd
    f = -gamma * ((bd * beta) + (hd * alpha) + fd)
    g = -gamma * ((hd * beta) + (ad * alpha) + gd)
    h = gamma**2 * hd
    u = gamma**2 * gd
    v = gamma**2 * fd
    w = -gamma * ((fd * beta) + (gd * alpha) + dd)
    
    ## Verification of (V.A.1)
    # a = 204.024
    # b = 225.000
    # c = 66.976
    # f = -177.452 / 2
    # g = -127.567 / 2
    # h = -102.452 / 2
    # u = -127.567 / 2
    # v = -177.452 / 2
    # w = 133.952 / 2
    # d = 66.976
    # r = 4
    
    # az^2+by^2+cz^2+2fyz+2gzx+2hxy+2ux+2vy+2wz+d = 0
    
    if debug:
        print(f"Cone parameters: a: {a}, b: {b}, c: {c}, 2f: {2*f} 2g: {2*g} 2h: {2*h} 2u: {2*u} 2v: {2*v} 2w: {2*w} d: {d}")

    ### (Verified from here except for projection and only for gamma=-1 (everything else fails catastrophically))

    # solve cubic equation for roots
    # (10)
    roots = np.roots([1, -(a + b + c), (b*c + c*a + a*b - f**2 - g**2 - h**2), -(a*b*c + 2*f*g*h - a*f**2 - b*g**2 - c*h**2)])
    roots = np.sort(roots.real)  # only real parts
    roots = roots[::-1]  # descending order
        
    Lam1 = roots[0]
    Lam2 = roots[1]
    Lam3 = roots[2]
    
    if debug:
        print(f"Roots: Lam1: {Lam1}, Lam2: {Lam2}, Lam3: {Lam3}")
        
    if Lam1 < Lam2:  # Doesn't happen? -> change root order?
        # (31)
        n = np.sqrt((Lam1 - Lam3) / (Lam2 - Lam3))
        m = np.sqrt((Lam2 - Lam1) / (Lam2 - Lam3))  # +-
        l = 0
        
        surface_normal_vectors_im = [[l, m, n, 1], [l, -m, n, 1]]  # Vectors wrt. image frame
    elif Lam1 > Lam2:
        # (32)
        n = np.sqrt((Lam2 - Lam3) / (Lam1 - Lam3))
        m = 0
        l = np.sqrt((Lam1 - Lam2) / (Lam1 - Lam3))  # +-
        
        surface_normal_vectors_im = [[l, m, n, 1], [-l, m, n, 1]]
    else:  # Circle case
        # (33)
        n = 1
        m = 0
        l = 0
        
        surface_normal_vectors_im = [[l, m, n, 1], [l, m, n, 1]]
        
    surface_normal_vectors_im = np.matrix(np.array(surface_normal_vectors_im)).T
    
    if debug:
        print(f"Surface normal vectors (image frame): {surface_normal_vectors_im}")
    
    # (12)
    mi = []
    li = []
    ni = []
    for root in roots:
        t1 = ((b - root) * g) - (f * h)
        t2 = ((a - root) * f) - (g * h)
        t3 = ((-(a - root) * (t1 / t2)) / g) - (h / g)
        
        if debug:
            print(f"t1: {t1}, t2: {t2}, t3: {t3}")
        
        curr_mi = 1 / np.sqrt(1 + (t1/t2)**2 + t3**2)
        curr_li = (t1/t2) * curr_mi
        curr_ni = t3 * curr_mi
        
        mi.append(curr_mi)
        li.append(curr_li)
        ni.append(curr_ni)
    
    li = np.array(li)
    mi = np.array(mi)
    ni = np.array(ni)
    
    crp = np.cross(li, mi)
    angle_between_crp_ni = angle_between(crp, ni)
    if angle_between_crp_ni > np.pi / 2:
        # Flip last column to ensure right-handed coordinate system
        li[2] = -li[2]
        mi[2] = -mi[2]
        ni[2] = -ni[2]
        if debug:
            print("li/mi/ni flipped to ensure right-handed coordinate system")
    
    # (8)
    T1 = np.matrix(np.array([[li[0], li[1], li[2], 0],
                            [mi[0], mi[1], mi[2], 0],
                            [ni[0], ni[1], ni[2], 0],
                            [0, 0, 0, 1]]))
    
    if debug:
        print(f"T1: {T1}")
    
    surface_normal_vectors_cam = T1 * surface_normal_vectors_im
            
    if debug:
        print(f"Surface normal vectors (camera frame): {surface_normal_vectors_cam}")
    
    # Estimate position of circle if r is given
    if r is not None:
        # (34)
        T0 = np.matrix(np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, focal_length_mm],
                                 [0, 0, 0, 1]]))
        
        if debug:
            print(f"T0: {T0}")
        
        # (14)
        T2 = np.matrix(np.array([[1, 0, 0, -(u*li[0] + v*mi[0] + w*ni[0]) / Lam1],
                                 [0, 1, 0, -(u*li[1] + v*mi[1] + w*ni[1]) / Lam2],
                                 [0, 0, 1, -(u*li[2] + v*mi[2] + w*ni[2]) / Lam3],
                                 [0, 0, 0, 1]]))
        
        if debug:
            print(f"T2: {T2}")
        
        # (19)
        T3s = []
        for i in range(surface_normal_vectors_im.shape[1]):
            l = surface_normal_vectors_im[0,i]
            m = surface_normal_vectors_im[1,i]
            n = surface_normal_vectors_im[2,i]
            T3s.append(np.matrix(np.array([[-m / np.sqrt(l**2 + m**2), -l*n / np.sqrt(l**2 + m**2), l, 0],
                                            [l / np.sqrt(l**2 + m**2), -m*n / np.sqrt(l**2 + m**2), m, 0],
                                            [0,                         np.sqrt(l**2 + m**2), n, 0],
                                            [0, 0, 0, 1]])))
            
        if debug:
            print(f"T3s: {T3s}")
        
        pupil_center_vectors_cam = np.matrix(np.zeros_like(surface_normal_vectors_cam))
        for i, T3 in enumerate(T3s):
            # (36) -> (38)
            A = (Lam1 * T3[0,0]**2) + (Lam2 * T3[1,0]**2) + (Lam3 * T3[2,0]**2)
            B = (Lam1 * T3[0,0] * T3[0,2]) + (Lam2 * T3[1,0] * T3[1,2]) + (Lam3 * T3[2,0] * T3[2,2])
            C = (Lam1 * T3[0,1] * T3[0,2]) + (Lam2 * T3[1,1] * T3[1,2]) + (Lam3 * T3[2,1] * T3[2,2])
            D = (Lam1 * T3[0,2]**2) + (Lam2 * T3[1,2]**2) + (Lam3 * T3[2,2]**2)
            
            if debug:
                print(f"A: {A}, B: {B}, C: {C}, D: {D}")
        
            # (41)
            Zdo = (A * r) / np.sqrt(B**2 + C**2 - (A * D))  # +-
            Xdo = -(B / A) * Zdo
            Ydo = -(C / A) * Zdo
            
            XYZdo = np.matrix(np.array([[Xdo, Ydo, Zdo, 1],
                                        [-Xdo, -Ydo, -Zdo, 1]])).T
            
            if debug:
                print(f"Circle center in local frame: {XYZdo}")
            
            xyzco = T0 * T1 * T2 * T3 * XYZdo
            
            if debug:
                print(f"Circle center in camera frame: {xyzco}")
            
            if debug:
                print(f"Total transform T: {T0 * T1 * T2 * T3}")
            
            if xyzco[2,0] > 0:  # select Zdo sign so that zco is positive
                pupil_center_vectors_cam[:,i] = (xyzco[:,0])
            else:
                pupil_center_vectors_cam[:,i] = (xyzco[:,1])
                
            # Swirski-specific adjustment: Gaze vectors need to point towards the camera
            if np.dot(pupil_center_vectors_cam[:-1,i].A1, surface_normal_vectors_cam[:-1,i].A1) > 0:  # pointing away from camera -> flip vector
                surface_normal_vectors_cam[:-1,i] = -surface_normal_vectors_cam[:-1,i]
                if debug:
                    print(f"Surface normal vector {i} flipped")
        
        if debug:
            print(f"Pupil center vectors (camera frame): {pupil_center_vectors_cam}")
            
    else:
        # TODO Probably wrong
        pupil_center_vectors_cam = [np.matrix(np.array([[ex, ey, -focal_length_mm, 1]])).T]
        T0 = None
        T2 = None
        T3 = None
        
    
    return surface_normal_vectors_im, surface_normal_vectors_cam, pupil_center_vectors_cam, (T0, T1, T2, T3s)      

def plot_surface_normal(surface_normal_vectors_im, surface_normal_vectors_cam):
    # plot normal vector in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, surface_normal_vectors_im[0], surface_normal_vectors_im[1], surface_normal_vectors_im[2], length=1, color='b')
    ax.quiver(0, 0, 0, surface_normal_vectors_cam[0], surface_normal_vectors_cam[1], surface_normal_vectors_cam[2], length=1, color='r')
    
    # plot sphere
    # Make data
    r = 1
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='linen', alpha=0.5)

    # plot circular curves over the surface
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.zeros(100)
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    ax.plot(x, y, z, color='black', alpha=0.75)
    ax.plot(z, x, y, color='black', alpha=0.75)

    ## add axis lines
    zeros = np.zeros(1000)
    line = np.linspace(-r,r,1000)

    ax.plot(line, zeros, zeros, color='black', alpha=0.75)
    ax.plot(zeros, line, zeros, color='black', alpha=0.75)
    ax.plot(zeros, zeros, line, color='black', alpha=0.75)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
def convert_ellipse_params_to_mm(pupil_ellipse, dimensions, sensor_dimensions_mm):
    unit_factor = sensor_dimensions_mm[0] / dimensions[0]  # pixels to mm
    unit_conv_pupil_ellipse = [[(pupil_ellipse[0][i] - (dimensions[i] / 2)) * unit_factor for i in range(2)], [pupil_ellipse[1][i] * unit_factor for i in range(2)], pupil_ellipse[2]]  # pixels to mm and centering
    return unit_conv_pupil_ellipse

def convert_ellipse_params_to_px(unit_conv_pupil_ellipse, dimensions, sensor_dimensions_mm):
    unit_factor = sensor_dimensions_mm[0] / dimensions[0]  # pixels to mm
    pupil_ellipse = [[(unit_conv_pupil_ellipse[0][i] / unit_factor) + (dimensions[i] / 2)  for i in range(2)], [unit_conv_pupil_ellipse[1][i] / unit_factor for i in range(2)], unit_conv_pupil_ellipse[2]]  # pixels to mm and centering
    return pupil_ellipse

def calculate_3d_circle_outline(center, normal_vector, r, steps=100):
    normal_vector = unit_vector(normal_vector)
    z_ax_vec = np.array([0, 0, 1])
    rot_ax = unit_vector(np.cross(z_ax_vec, normal_vector))
    rot_vec = rot_ax * np.acos(np.dot(z_ax_vec, normal_vector))
    rot_mat = R.from_rotvec(rot_vec).as_matrix()
    
    circle_theta = np.linspace(0, 2*np.pi, steps)
    circle_points = np.matrix([np.cos(circle_theta) * r,
                    np.sin(circle_theta) * r,
                    np.zeros_like(circle_theta)])
    
    circle_points_transf = rot_mat * circle_points
    circle_points_transf += np.matrix(center).T
    
    # Returns Matrix with cols = point vectors
    return circle_points_transf

def calculate_swirski_for_frame(likely_pupil_ellipse, dimensions, debug_plot=False, sensor_focal_length_mm=1, sensor_dimensions_mm=[1,1], initial_pupil_radius_mm=2):
    # dimensions: [width, height] in pixels
    #sensor_dimensions_mm = [3.6, 3.6*(dimensions[1]/dimensions[0])]  # adjust for aspect ratio of dataset images
    #sensor_dimensions_mm = [3.6, 2.7]
    #sensor_dimensions_mm = np.array([1, 1])
    #sensor_dimensions_mm = [0.08413290156921527*1000, 0.08413290156921527*(dimensions[1]/dimensions[0])*1000]  # adjust for aspect ratio of dataset images  # TODO leads to interesting edge case (vector pointing too far up)
    #initial_pupil_radius_mm = 4/2  # average adult pupil diameter in mm

    unit_conv_likely_pupil_ellipse = convert_ellipse_params_to_mm(likely_pupil_ellipse, dimensions, sensor_dimensions_mm)

    # Run Cone unprojection algorithm
    surface_normal_vectors_im, surface_normal_vectors_cam, pupil_center_vectors_cam, transf_mat = calculate_surface_normal(unit_conv_likely_pupil_ellipse, sensor_focal_length_mm, initial_pupil_radius_mm, debug=debug_plot)
    # All coordinates w.r.t. camera center at (0,0,0)
    
    # Project normal vectors back to 2d image plane using perspective projection
    pupil_center_vectors_cam_proj = []
    surface_normal_vectors_cam_proj = []
    for i in range(pupil_center_vectors_cam.shape[1]):
        pupil_center_vectors_cam_proj.append(perspective_project(pupil_center_vectors_cam[:-1,i].A1, sensor_focal_length_mm))
        snv_end_points = pupil_center_vectors_cam[:-1,i] + surface_normal_vectors_cam[:-1,i]
        surface_normal_vectors_cam_proj.append(perspective_project(snv_end_points.A1, sensor_focal_length_mm))
        
    swirski_results = {
        "pupil_center_vectors_cam": pupil_center_vectors_cam,
        "surface_normal_vectors_cam": surface_normal_vectors_cam,
        "pupil_center_vectors_cam_proj": pupil_center_vectors_cam_proj,
        "surface_normal_vectors_cam_proj_endpoint": surface_normal_vectors_cam_proj,
        "unit_conv_likely_pupil_ellipse": unit_conv_likely_pupil_ellipse
    }

    if debug_plot:
        print(f"Unit converted likely pupil ellipse (mm): {unit_conv_likely_pupil_ellipse}")
        print(f"Normal vectors in image coordinates:\n {surface_normal_vectors_im}")
        print(f"Normal vectors in camera coordinates:\n {surface_normal_vectors_cam}")
        print(f"Pupil center vectors in camera coordinates:\n {pupil_center_vectors_cam}")
        
        print(f"Pupil center vectors projected to image plane:\n {pupil_center_vectors_cam_proj}")
        print(f"Surface normal vectors projected to image plane:\n {surface_normal_vectors_cam_proj}")
        
        el_cntr = unit_conv_likely_pupil_ellipse[0]
        
        test = transf_mat[1] * np.matrix(np.array([[el_cntr[0], el_cntr[1], -sensor_focal_length_mm, 1]]).T)
        print(test)

        camera_inv_vector = np.array([0, 0, 1])
        for i in range(surface_normal_vectors_cam.shape[1]):
            print(f"surface angle {i}: {np.rad2deg(angle_between(camera_inv_vector, surface_normal_vectors_cam[:-1,i].A1))}")
            #plot_surface_normal(surface_normal_vectors_im[:,i].A1, surface_normal_vectors_cam[:,i].A1)
                    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # plot normal vector in 2d image plane as line
        ax.plot([el_cntr[0], el_cntr[0] + surface_normal_vectors_im[0,0]], [el_cntr[1], el_cntr[1] + surface_normal_vectors_im[1,0]], c='b')
        ax.plot([el_cntr[0], el_cntr[0] + surface_normal_vectors_im[0,1]], [el_cntr[1], el_cntr[1] + surface_normal_vectors_im[1,1]], c='b')
        ax.plot([el_cntr[0], el_cntr[0] + surface_normal_vectors_cam[0,0]], [el_cntr[1], el_cntr[1] + surface_normal_vectors_cam[1,0]], c='r')
        ax.plot([el_cntr[0], el_cntr[0] + surface_normal_vectors_cam[0,1]], [el_cntr[1], el_cntr[1] + surface_normal_vectors_cam[1,1]], c='r')
        ax.plot([pupil_center_vectors_cam_proj[0][0], surface_normal_vectors_cam_proj[0][0]], [pupil_center_vectors_cam_proj[0][1], surface_normal_vectors_cam_proj[0][1]], c='g')
        ax.plot([pupil_center_vectors_cam_proj[1][0], surface_normal_vectors_cam_proj[1][0]], [pupil_center_vectors_cam_proj[1][1], surface_normal_vectors_cam_proj[1][1]], c='y')
        ax.scatter([pupil_center_vectors_cam_proj[0][0]], [pupil_center_vectors_cam_proj[0][1]], c='g')
        ax.scatter([pupil_center_vectors_cam_proj[1][0]], [pupil_center_vectors_cam_proj[1][1]], c='y')
        # plot ellipse
        ellipse_patch = patches.Ellipse((unit_conv_likely_pupil_ellipse[0][0], unit_conv_likely_pupil_ellipse[0][1]), unit_conv_likely_pupil_ellipse[1][0], unit_conv_likely_pupil_ellipse[1][1], angle=unit_conv_likely_pupil_ellipse[2], fill=False, edgecolor='r')
        ax.add_patch(ellipse_patch)
        
        # plot ellipse axes
        ellipse_angle_rad = np.deg2rad(unit_conv_likely_pupil_ellipse[2])
        axis1_dir = np.array([np.cos(ellipse_angle_rad), np.sin(ellipse_angle_rad)])
        axis2_dir = np.array([-np.sin(ellipse_angle_rad), np.cos(ellipse_angle_rad)])
        axis1_start = np.array([unit_conv_likely_pupil_ellipse[0][0], unit_conv_likely_pupil_ellipse[0][1]]) - (axis1_dir * (unit_conv_likely_pupil_ellipse[1][0] / 2))
        axis1_end = np.array([unit_conv_likely_pupil_ellipse[0][0], unit_conv_likely_pupil_ellipse[0][1]]) + (axis1_dir * (unit_conv_likely_pupil_ellipse[1][0] / 2))
        axis2_start = np.array([unit_conv_likely_pupil_ellipse[0][0], unit_conv_likely_pupil_ellipse[0][1]]) - (axis2_dir * (unit_conv_likely_pupil_ellipse[1][1] / 2))
        axis2_end = np.array([unit_conv_likely_pupil_ellipse[0][0], unit_conv_likely_pupil_ellipse[0][1]]) + (axis2_dir * (unit_conv_likely_pupil_ellipse[1][1] / 2))
        ax.plot([axis1_start[0], axis1_end[0]], [axis1_start[1], axis1_end[1]], c='r', linestyle='--')
        ax.plot([axis2_start[0], axis2_end[0]], [axis2_start[1], axis2_end[1]], c='g', linestyle='--')

        #ax.set_xlim([-1, 1])
        #ax.set_ylim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', 'box')
        plt.grid()

        plt.show()
        
        circle_points_3d = []
        circle_centers_3d = []
        circle_normal_vectors = []
        circle_colors = ["r", "g", "b", "orange"]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('persp', focal_length=1)
        #ax.view_init(elev=60, azim=-60, roll=0)
        
        for i in range(surface_normal_vectors_cam.shape[1]):
            circle_normal = surface_normal_vectors_cam[:-1,i].A1
            circle_center = pupil_center_vectors_cam[:-1,i].A1
            circle_radius = initial_pupil_radius_mm
            
            circle_points_transf = calculate_3d_circle_outline(circle_center, circle_normal, circle_radius, 100)
            
            circle_points_3d.append(circle_points_transf)
            circle_centers_3d.append(circle_center)
            circle_normal_vectors.append(circle_normal)
        
        for i in range(len(circle_points_3d)):
            ax.plot(circle_points_3d[i][0,:], circle_points_3d[i][1,:], circle_points_3d[i][2,:], color=circle_colors[i])
            ax.scatter([circle_centers_3d[i][0]], circle_centers_3d[i][1], [circle_centers_3d[i][2]], c=circle_colors[i], s=20)
            ax.quiver(circle_centers_3d[i][0], circle_centers_3d[i][1], circle_centers_3d[i][2], circle_normal_vectors[i][0], circle_normal_vectors[i][1], circle_normal_vectors[i][2], length=1, color=circle_colors[i])

        #ax.scatter([0], [0], [0], c='lime', s=50)
        
        #ax.set_xlim([-6, 6])
        #ax.set_ylim([-6, 6])
        #ax.set_zlim([12, 0])
        ax.set_aspect('equal', 'box', 'C')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.grid()
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for i in range(len(circle_points_3d)):
            circle_points_projected = np.array([perspective_project(circle_points_3d[i][:,j].A1, sensor_focal_length_mm) for j in range(circle_points_3d[i].shape[1])])
            ax.plot(circle_points_projected[:,0], circle_points_projected[:,1], color=circle_colors[i])
            circle_center_projected = perspective_project(circle_centers_3d[i], sensor_focal_length_mm)
            ax.scatter([circle_center_projected[0]], [circle_center_projected[1]], color=circle_colors[i], s=20)
            circle_normal_vector_projected = perspective_project(circle_centers_3d[i] + circle_normal_vectors[i], sensor_focal_length_mm)
            ax.plot([circle_center_projected[0], circle_normal_vector_projected[0]], [circle_center_projected[1], circle_normal_vector_projected[1]], color=circle_colors[i])
            print(circle_normal_vector_projected)
            print(circle_normal_vectors)
            print(circle_centers_3d[i] + circle_normal_vectors[i])
            
        ellipse_patch = patches.Ellipse((unit_conv_likely_pupil_ellipse[0][0], unit_conv_likely_pupil_ellipse[0][1]), unit_conv_likely_pupil_ellipse[1][0], unit_conv_likely_pupil_ellipse[1][1], angle=unit_conv_likely_pupil_ellipse[2], fill=False, edgecolor='k')
        ax.add_patch(ellipse_patch)
        
        plt.grid()
        plt.show()
        
        
    return swirski_results

def lsq_intersection_of_lines(point_vecs, normal_vecs):
    # TODO idea: use weighted sum depending on ellipse fit quality etc.?
    
    vec_len = len(point_vecs[0])
    
    S1 = np.matrix(np.zeros((vec_len, vec_len)))
    S2 = np.matrix(np.zeros((vec_len, 1)))
    
    for i in range(len(normal_vecs)):
        nv = np.matrix(normal_vecs[i]).T
        pv = np.matrix(point_vecs[i]).T
        
        nv = unit_vector(nv)
        
        s1 = np.identity(vec_len) - (nv * nv.T)
        s2 = (np.identity(vec_len) - (nv * nv.T)) * pv
        
        S1 += s1
        S2 += s2
    
    c_proj = np.linalg.pinv(S1) * S2
    
    return c_proj.A1

def swirski_for_series(frames, likely_pupil_ellipses, focal_length_mm, sensor_dimensions_mm, initial_pupil_radius_mm, estimated_sphere_distance=40):
    def resolve_projection_ambiguities(p_proj, n_proj, c_proj):
            
        ang_gaze_c_proj = angle_between(c_proj - p_proj[0], n_proj[0])
        
        if ang_gaze_c_proj > np.deg2rad(90):
            return 0
        else:
            return 1
        
    ### Model Initialization

    # Sensor parameters
    dimensions = [frames[0].shape[1], frames[0].shape[0]]

    swirskis = [calculate_swirski_for_frame(likely_pupil_ellipses[i], dimensions, False, focal_length_mm, sensor_dimensions_mm, initial_pupil_radius_mm) for i in range(len(likely_pupil_ellipses))]

    p_proj_all = np.array([swirski["pupil_center_vectors_cam_proj"] for swirski in swirskis])
    n_proj_all = np.array([np.array(swirski["surface_normal_vectors_cam_proj_endpoint"]) - np.array(swirski["pupil_center_vectors_cam_proj"]) for swirski in swirskis])
    p_all = np.array([swirski["pupil_center_vectors_cam"] for swirski in swirskis])
    n_all = np.array([swirski["surface_normal_vectors_cam"] for swirski in swirskis])
    r_all = np.array([initial_pupil_radius_mm for _ in swirskis])

    sphere_c_proj = lsq_intersection_of_lines(p_proj_all[:,0,:], n_proj_all[:,0,:])
    fixed_z_mm = estimated_sphere_distance  # initial estimate
    sphere_c = perspective_unproject(sphere_c_proj, fixed_z_mm, focal_length_mm)

    resolved_idxs = np.array([resolve_projection_ambiguities(p_proj, n_proj, sphere_c_proj) for p_proj, n_proj in zip(p_proj_all, n_proj_all)])
    p_proj_all_filtered = np.array([p_proj_all[idx][resolved_idx] for idx, resolved_idx in enumerate(resolved_idxs)])
    n_proj_all_filtered = np.array([n_proj_all[idx][resolved_idx] for idx, resolved_idx in enumerate(resolved_idxs)])
    p_all_filtered = np.array([p_all[idx][:,resolved_idx] for idx, resolved_idx in enumerate(resolved_idxs)])
    n_all_filtered = np.array([n_all[idx][:,resolved_idx] for idx, resolved_idx in enumerate(resolved_idxs)])
    
    p_hats = []
    Rs = []

    for p_i, n_i in zip(p_all_filtered, n_all_filtered):
        p_i_hat = lsq_intersection_of_lines([sphere_c, p_i[:-1]], [n_i[:-1], -p_i[:-1]])
        R_i = np.linalg.norm(sphere_c - p_i_hat)
        
        p_hats.append(p_i_hat)
        Rs.append(R_i)
        
    sphere_R_mean = np.mean(np.array(Rs))  # (9)
    
    print(f"Sphere radius estimate: {sphere_R_mean:.2f}mm")
    
    # Ignore R guess and fix to 24mm
    
    sphere_R_mean = 12

    p_dashs = []
    n_dashs = []
    r_dashs = []
    inlier_idxs = []

    for i, (p_i, r_i) in enumerate(zip(p_all_filtered, r_all)):
        sphere = Sphere(sphere_c, sphere_R_mean)
        line = Line(p_i[:-1], p_i[:-1])
        try:
            point_a, point_b = sphere.intersect_line(line)  # (10)
        except:
            # Outlier solution, line does not intersect sphere
            continue
        
        if point_a[2] < point_b[2]:  # select point closer to camera / 0
            p_i_dash = point_a
        else:
            p_i_dash = point_b
        
        n_i_dash = (p_i_dash - sphere_c) / sphere_R_mean  # (11)
        r_i_dash = r_i * (p_i_dash[2] / p_i[2])  # (12)
        
        p_dashs.append(p_i_dash)
        n_dashs.append(n_i_dash)
        r_dashs.append(r_i_dash)
        inlier_idxs.append(i)
    
    print(len(inlier_idxs), len(p_all_filtered))
    
    frames_filtered = []
    likely_pupil_ellipses_filtered = []
    for i in inlier_idxs:
        frames_filtered.append(frames[i])
        likely_pupil_ellipses_filtered.append(likely_pupil_ellipses[i])
                
    #### Model Optimisation

    def metric_outline_contrast_for_frame(frame, pupil_ellipse):
        
        frame = frame - 230  # TODO hardcoded visualization fix
        
        # Transform to polar view around ellipse center
        pol_angular_size = 128
        longer_side = np.max(frame.shape)
        frame_polar = cv2.warpPolar(frame, (longer_side, pol_angular_size), pupil_ellipse[0], longer_side, flags=cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS)  # Not including WARP_FILL_OUTLIERS causes non-deterministic bugs (overflows)
        
        # Warp so that ellipse outline becomes straight
        eccentricity = np.sqrt(1 - ((pupil_ellipse[1][1]**2) / (pupil_ellipse[1][0]**2)))
        mean_radius = np.mean(pupil_ellipse[1]) / 2
        roll = int(pupil_ellipse[2] / 360 * pol_angular_size)  # based on angle
        ell_b_itr = np.arange(0,frame_polar.shape[1],1)  # Pupil diameter = y coord
        ell_warp_map_itr = np.linspace(0, 2*np.pi, frame_polar.shape[0])  # Angle
        ell_warp_mgr = np.array(np.meshgrid(ell_warp_map_itr, ell_b_itr, indexing='ij', copy=False))
        ell_warp_maps = np.array(ell_warp_mgr[1] / np.sqrt(1 - (eccentricity * np.cos((ell_warp_mgr[0])))**2) - ell_warp_mgr[1], dtype=np.float32)
        frame_polar_warped = cv2.remap(frame_polar, np.roll(ell_warp_maps, roll, axis=0), np.zeros_like(frame_polar, dtype=np.float32), interpolation=cv2.INTER_LINEAR | cv2.WARP_RELATIVE_MAP)
        
        # plt.figure()
        # ax = plt.gca()
        # ax.imshow(np.swapaxes(frame_polar_warped,0,1), cmap='gray')
        # ax.set_aspect(frame_polar_warped.shape[0] / frame_polar_warped.shape[1])
        # plt.show()
        
        # (24)
        def smootherstep(t, eps=.5):
            # eps = px
            if t >= eps:
                return 1
            elif t <= -eps:
                return 0
            else:
                return 6*((t+eps)/(2*eps))**5 - 15*((t+eps)/(2*eps))**4 + 10*((t+eps)/(2*eps))**3
        
        band_width_px = 5
        
        weights_in_row = np.array([smootherstep(i - (mean_radius - band_width_px)) - smootherstep(i - mean_radius) for i in range(frame_polar_warped.shape[1])])
        weights_out_row = np.array([smootherstep(i - mean_radius) - smootherstep(i - (mean_radius + band_width_px)) for i in range(frame_polar_warped.shape[1])])
        
        weights_in = np.tile(weights_in_row, (frame_polar_warped.shape[0], 1))
        weights_out = np.tile(weights_out_row, (frame_polar_warped.shape[0], 1))
        
        mean_in = np.average(frame_polar_warped, weights=weights_in)
        mean_out = np.average(frame_polar_warped, weights=weights_out)
        
        # center_int = (int(likely_pupil_ellipse[0][0]), int(likely_pupil_ellipse[0][1]))
        # axes_int = (int(likely_pupil_ellipse[1][0]//2), int(likely_pupil_ellipse[1][1]//2))
        # angle_int = int(likely_pupil_ellipse[2])
        # Note that this introduces discretization issues!
        # brightnesses = np.array([calculate_outline_contrast(frame, center_int, p, band_width_px) for p in cv2.ellipse2Poly(center_int, axes_int, angle_int, 0, 360, 10)])
        # means = np.mean(brightnesses, axis=0)
        
        E = mean_out - mean_in  # (22)
        
        return E

    def error_fun(x, sphere_R_mean, frames, focal_length_mm, sensor_dimensions_mm, debug_plot=False):
        sphere_c = np.array([x[0], x[1], x[2]])
        dimensions = [frames[0].shape[1], frames[0].shape[0]]
        
        total_loss = 0
        
        total_frame_count = len(x)//3-1
        max_debug_frames = 32
        plot_modulo = total_frame_count // max_debug_frames
        
        for i in np.arange(3, len(x), 3):
            frame_idx = i//3-1
            
            angles = np.array([x[i], x[i+1], 0])
            r_dash = x[i+2]
            
            # Convert theta, phi back to rotation matrix
            rot = R.from_euler("yxz", angles, degrees=True)
            rot_mat = rot.as_matrix()
            normal_vector = np.matrix(np.array([0,0,-1])).T
            
            n_dash = rot_mat * normal_vector
            p_dash = np.matrix(sphere_c).T + (n_dash * sphere_R_mean)  #  TODO ???? scaling factor missing in paper, should we just assume fixed sphere radius?
            # Note that this also causes issues if the sphere intersects the camera center -> circle crosses image plane
            
            # Project circle to image plane and calculate outline contrast
            # Solve ellipse for five projected points - there is likely a closed-form solution similar to the surface normal calculation above.
            circle_theta = np.linspace(0, 2*np.pi, 6)[:-1]  # Full circle, last p == first p
            circle_points = np.matrix([np.cos(circle_theta) * r_dash,
                            np.sin(circle_theta) * r_dash,
                            np.zeros_like(circle_theta)])
            
            circle_points_transf = (rot_mat * circle_points) + p_dash
            circle_points_proj = np.array([perspective_project(circle_points_transf[:,j].A1, focal_length_mm) for j in range(circle_points_transf.shape[1])])
            
            # https://math.stackexchange.com/questions/163920/how-to-find-an-ellipse-given-five-points
            dlt_mat = np.matrix(np.array([[p**2, p*q, q**2, p, q, 1] for (p, q) in circle_points_proj]))
            
            ## Find nullspace via SVD (issues with number precision)
            #U, S, Vh = np.linalg.svd(dlt_mat)
            #V = Vh.T
            #zero_cols = [i for i in range(S.shape[0]) if S[i] < 1e-2]  # find zero values with tolerance
            #ellipse_params = V[:,zero_cols[0]].A1
            
            ## Find nullspace via sympy
            ellipse_params = np.matrix(sympy.Matrix(dlt_mat).nullspace()[0], dtype=np.float64).A1
            
            def convert_ellipse_general_to_canonical(general_ellipse_params):  # Assumes Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
                # Convert to ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0
                a = general_ellipse_params[0]
                b = general_ellipse_params[1]/2
                c = general_ellipse_params[2]
                d = general_ellipse_params[3]/2
                f = general_ellipse_params[4]/2
                g = general_ellipse_params[5]
                
                # From https://mathworld.wolfram.com/Ellipse.html
                ea = 2 * np.sqrt((2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)) / ((b**2 - a*c) * (np.sqrt((a-c)**2 + 4*b**2) - (a+c))))  # 2* -> diameter
                eb = 2 * np.sqrt((2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)) / ((b**2 - a*c) * (- np.sqrt((a-c)**2 + 4*b**2) - (a+c))))
                ex = (c*d - b*f) / (b**2 - a*c)
                ey = (a*f - b*d) / (b**2 - a*c)
                etheta = np.rad2deg((1/2) * np.atan2(-2*b, c-a))  # issues with exact angles
                
                ellipse = ((ex, ey), (ea, eb), etheta)
                
                return ellipse
            
            ellipse = convert_ellipse_general_to_canonical(ellipse_params)
            
            sphere_c_reproj = perspective_project(sphere_c, focal_length_mm)
            
            if debug_plot and ((frame_idx % plot_modulo) == 0):
                plt.figure()
                ax = plt.gca()
                ax.set_xlim([-3, 3])
                ax.set_ylim([3, -3])
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_aspect('equal', 'box')
                ax.set_title(f"Frame #{frame_idx}")
                plt.grid()
                
                ax.imshow(frames_filtered[frame_idx], extent=[-sensor_dimensions_mm[0]/2, sensor_dimensions_mm[0]/2, -sensor_dimensions_mm[1]/2, sensor_dimensions_mm[1]/2], alpha=1, cmap='gray', origin='lower')
                
                ax.scatter(circle_points_proj[:,0], circle_points_proj[:,1], c='g', s=5)
                
                # Projection check
                circle_theta_2 = np.linspace(0, 2*np.pi, 100)
                circle_points_2 = np.matrix([np.cos(circle_theta_2) * r_dash,
                                np.sin(circle_theta_2) * r_dash,
                                np.zeros_like(circle_theta_2)])
                
                circle_points_transf_2 = (rot_mat * circle_points_2) + p_dash
                circle_points_proj_2 = np.array([perspective_project(circle_points_transf_2[:,j].A1, focal_length_mm) for j in range(circle_points_transf_2.shape[1])])
                
                ax.scatter(circle_points_proj_2[:,0], circle_points_proj_2[:,1], c='y', s=.2)
                
                # plot ellipse
                ellipse_patch = patches.Ellipse((ellipse[0][0], ellipse[0][1]), ellipse[1][0], ellipse[1][1], angle=ellipse[2], fill=False, edgecolor='b', linewidth=1)
                ax.add_patch(ellipse_patch)
                
                gt_ellipse = convert_ellipse_params_to_mm(likely_pupil_ellipses_filtered[frame_idx], dimensions, sensor_dimensions_mm)
                ellipse_patch_gt = patches.Ellipse((gt_ellipse[0][0], gt_ellipse[0][1]), gt_ellipse[1][0], gt_ellipse[1][1], angle=gt_ellipse[2], fill=False, edgecolor='r', linewidth=1)
                ax.add_patch(ellipse_patch_gt)
                
                circle_points_3d = calculate_3d_circle_outline(sphere_c, sphere_c, sphere_R_mean)
                circle_points_projected = np.array([perspective_project(circle_points_3d[:,j].A1, focal_length_mm) for j in range(circle_points_3d.shape[1])])
                ax.plot(circle_points_projected[:,0], circle_points_projected[:,1], color='g')
                
                ax.scatter([sphere_c_proj[0]], [sphere_c_proj[1]], c='lime', s=50, zorder=10)
                ax.scatter([sphere_c_reproj[0]], [sphere_c_reproj[1]], c='g', s=25, zorder=10)
                
                plt.show()
            
            ellipse_px = convert_ellipse_params_to_px(ellipse, dimensions, sensor_dimensions_mm)
            frame_loss = metric_outline_contrast_for_frame(frames[frame_idx], ellipse_px)
            
            total_loss += frame_loss
        
        total_loss = -total_loss  # minimization problem
        print(total_loss)
        
        return total_loss

    x0 = [sphere_c[0], sphere_c[1], sphere_c[2]]
    for i in range(len(p_dashs)):
        # Convert to theta, phi
        rot_mat = rotation_matrix_from_vectors(np.array([0,0,-1]), n_dashs[i])
        rot =  R.from_matrix(rot_mat)
        angles = rot.as_euler("yxz", degrees=True)
        
        x0.append(angles[0])
        x0.append(angles[1])
        x0.append(r_dashs[i])

    error_fun(x0, sphere_R_mean, frames_filtered, focal_length_mm, sensor_dimensions_mm, debug_plot=False)
    
    opt_res = scipy.optimize.minimize(error_fun, x0, args=(sphere_R_mean, frames_filtered, focal_length_mm, sensor_dimensions_mm), method='BFGS')
    opt_x = opt_res.x
    
    print(x0)
    print(opt_x)
    error_fun(opt_x, sphere_R_mean, frames_filtered, focal_length_mm, sensor_dimensions_mm, debug_plot=True)

    plt.figure()
    ax = plt.gca()
    ax.set_xlim([-3, 3])
    ax.set_ylim([3, -3])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal', 'box')
    plt.grid()
    
    ax.imshow(frames_filtered[0], extent=[-sensor_dimensions_mm[0]/2, sensor_dimensions_mm[0]/2, -sensor_dimensions_mm[1]/2, sensor_dimensions_mm[1]/2], alpha=1, cmap='gray', origin='lower')

    for i, swirski in enumerate(swirskis):
        p_proj = swirski["pupil_center_vectors_cam_proj"]
        n_proj = swirski["surface_normal_vectors_cam_proj_endpoint"]
        unit_conv_likely_pupil_ellipse = swirski["unit_conv_likely_pupil_ellipse"]
        
        # Plot normal vector in 2d image plane as line
        ax.axline([p_proj[0][0], p_proj[0][1]], [n_proj[0][0], n_proj[0][1]], c='b', linewidth=.1)
        color = 'b'
        if i not in inlier_idxs:
            color = 'r'
        ax.scatter([p_proj[0][0]], [p_proj[0][1]], c=color, s=5)
        
        # plot ellipse
        ellipse_patch = patches.Ellipse((unit_conv_likely_pupil_ellipse[0][0], unit_conv_likely_pupil_ellipse[0][1]), unit_conv_likely_pupil_ellipse[1][0], unit_conv_likely_pupil_ellipse[1][1], angle=unit_conv_likely_pupil_ellipse[2], fill=False, edgecolor='r', linewidth=.1)
        ax.add_patch(ellipse_patch)

    ax.scatter([sphere_c_proj[0]], [sphere_c_proj[1]], c='lime', s=50, zorder=10)
    
    plt.show()

    #TODO simulate circle
    #TODO align vector format!!
