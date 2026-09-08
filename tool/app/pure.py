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

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(9, la)
       
def automatic_brightness_and_contrast(image: np.ndarray, clip_hist_percent: float = .1):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_size = len(hist)
    accumulator = [float(hist[0])]
    for i in range(1, hist_size):
        accumulator.append(accumulator[i - 1] + float(hist[i]))
    maximum = accumulator[-1]
    clip = clip_hist_percent * (maximum / 100.0) / 2.0

    minimum_gray, maximum_gray = 0, hist_size - 1
    while minimum_gray < hist_size and accumulator[minimum_gray] < clip:
        minimum_gray += 1
    while maximum_gray > 0 and accumulator[maximum_gray] >= (maximum - clip):
        maximum_gray -= 1

    if maximum_gray == minimum_gray:
        # Fallback: keine sinnvolle Spreizung möglich
        return image.copy(), 1.0, 0.0

    alpha = 255.0 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta

def morphological_filter_match(arr, kernel): # kernel format: 1 = 1, 0 = wildcard, -1 = 0
    # mirror kernel along both axes because convolution does the same
    kernel = np.flip(np.flip(kernel,0),1)
    kernel_sum = np.sum(np.clip(kernel,0,1))
    conv_res = scipy.signal.convolve2d(arr, kernel, mode="valid")
    conv_res = np.pad(conv_res, ((kernel.shape[0]//2, kernel.shape[0]//2), (kernel.shape[1]//2, kernel.shape[1]//2)), mode='constant', constant_values=0)
    return conv_res > kernel_sum-.5  # issues with number precision

def morphological_filter_replace(arr, conv_res, kernel): # kernel format: 1 = 1, 0 = wildcard, -1 = 0
    ksx = kernel.shape[0]//2
    ksy = kernel.shape[1]//2
    # get indexes of nonzero in conv_res
    idxs = np.nonzero(conv_res)
    for i in range(len(idxs[0])):
        arr[idxs[0][i]-ksx:idxs[0][i]+ksx+1, idxs[1][i]-ksy:idxs[1][i]+ksy+1] = np.clip(arr[idxs[0][i]-ksx:idxs[0][i]+ksx+1, idxs[1][i]-ksy:idxs[1][i]+ksy+1] + kernel, 0, 1)
    return arr

def else_morphological_filter(edges):
    patterns_a_thinning = [np.array([[0, 0, -1],
                                     [1, 1, 0],
                                     [0, 1, 0]]),
                           np.array([[0, 0, 0],
                                     [0, -1, 0],
                                     [0, 0, 0]])
                           ]
    
    
    patterns_b_straighten = [np.array([[0, 1, 0],
                                       [1, -1, 1],
                                       [0, 0, 0]]),
                             np.array([[0, -1, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]])
                             ]
    
    patterns_c_straighten = [np.array([[0, 1, 1, 0, 0],
                                       [1, -1, -1, 1, 0],
                                       [0, 0, 0, 0, 0]]),
                             np.array([[0, -1, -1, 0, 0],
                                       [0, 1, 1, 0, 0],
                                       [0, 0, 0, 0, 0]])
                             ]
    
    
    patterns_d_ortho = [np.array([[1, 1, 0],
                                  [0, 0, 1],
                                  [0, 0, 1]]),
                        np.array([[0, -1, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]])
                        ]
    
    
    patterns_e_ortho = [np.array([[1, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 1]]),
                        np.array([[0, 0, 0, 0, 0],
                                  [0, 0, 0, -1, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]])
                        ]
                        
    patterns_f_ortho = [np.array([[0, 0, 1, 0, 0],
                                  [0, 1, 0, 1, 0],
                                  [1, 0, 0, 0, 1]]),
                        np.array([[0, 0, -1, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]])
                        ]
    
    
    patterns_g_ortho = [np.array([[0, 0, 1, 1, 0, 0, 0],
                                  [0, 1, 0, 0, 1, 0, 0],
                                  [1, 0, 0, 0, 0, 1, 0]]),
                        np.array([[0, 0, 0, -1, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0]])
                        ]
    
    patterns = [
        patterns_a_thinning,
        patterns_b_straighten,
        patterns_c_straighten,
        patterns_d_ortho,
        patterns_e_ortho,
        patterns_f_ortho,
        patterns_g_ortho
    ]
    for pattern in patterns:
        pattern_search = pattern[0]
        pattern_replace = pattern[1]
        
        for k in range(4):
            curr_pattern_search = np.rot90(pattern_search, k)
            curr_pattern_replace = np.rot90(pattern_replace, k)
            
            match_arr = morphological_filter_match(edges, curr_pattern_search)
            edges = morphological_filter_replace(edges, match_arr, curr_pattern_replace)
    
    return edges

def calculate_chains_for_frame(edges):
    def calculate_chain_for_px(edges, edges_done, p_start):
        def check_all_dirs(edges, curr_p, curr_chain):
            dirs_list = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
            for idx, dir in enumerate(dirs_list):
                curr_dir_p = (curr_p[0]+dir[0], curr_p[1]+dir[1])
                if not (curr_dir_p[0] < 0 or curr_dir_p[1] < 0 or curr_dir_p[0] >= edges.shape[0] or curr_dir_p[1] >= edges.shape[1]):
                    if curr_dir_p not in curr_chain:
                        if edges[curr_dir_p] == 1:
                            return curr_dir_p  # segment continuation found
            return None
        
        curr_chain = [p_start]
        edges_done[p_start] = 1
        
        for i in range(2):  # go in both directions
            curr_p = p_start
            cnt = 0
            while True:
                curr_dir_p = check_all_dirs(edges, curr_p, curr_chain)
                if curr_dir_p is not None:
                    curr_chain.append(curr_dir_p)
                    curr_p = curr_dir_p
                    edges_done[curr_p] = len(curr_chain)
                else:
                    if cnt == 0:
                        cnt += 1
                        # reverse chain so start point is in the middle
                        curr_chain = curr_chain[::-1]
                    break
        
        return curr_chain
    
    edges_done = np.zeros_like(edges)
    p_cand_idxs = np.nonzero(edges)
    segments = []
    
    for i in range(len(p_cand_idxs[0])):
        curr_p_start = (p_cand_idxs[0][i], p_cand_idxs[1][i])
        if edges[curr_p_start] == 1 and edges_done[curr_p_start] == 0:  # segment available and not already visited
            curr_chain = calculate_chain_for_px(edges, edges_done, curr_p_start)
            segments.append(curr_chain)
            
    return segments

def calculate_avg_intensity_along_line(frame, p1, p2):
    # Get line points
    line_pts = list(zip(*skimage.draw.line(int(p1[1]), int(p1[0]), int(p2[1]), int(p2[0]))))
    # Throw out points outside image
    line_pts = [p for p in line_pts if p[0] >= 0 and p[0] < frame.shape[0] and p[1] >= 0 and p[1] < frame.shape[1]]
    if len(line_pts) == 0:
        return 0
    return np.mean([frame[p] for p in line_pts])

def calculate_outline_contrast(frame, p_center, p, length):
    dir_vec = p - p_center
    if np.linalg.norm(dir_vec) == 0:  # might happen due to too small ellipses
        return 0
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    p_in = p - dir_vec * length
    p_out = p + dir_vec * length
    inten_in = calculate_avg_intensity_along_line(frame, p, p_in)
    inten_out = calculate_avg_intensity_along_line(frame, p, p_out)
    
    return inten_in, inten_out

def calculate_outline_contrast_dig(frame, p_center, p, length):
    inten_in, inten_out = calculate_outline_contrast(frame, p_center, p, length)
    
    if inten_in < inten_out:
        return 1
    else:
        return 0

def pupil_detection(frame, canny_params=[50, 200], debug_plot=False):
    
    ## 3.1 Preprocessing
    
    #frame = cv2.equalizeHist(frame)
    # min-max normalization
    frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
    frame = frame.astype(np.uint8)
    
    ## 3.2 Edge Detection and Morphological Manipulation
    
    edges = cv2.Canny(frame, canny_params[0], canny_params[1], apertureSize=3) # 50 150
    edges = np.clip(edges, 0, 1)
    edges_morph_filtered = else_morphological_filter(edges.copy())
    #tc89(edges_morph_filtered)
    
    ## 3.3 Edge Segment Selection
    
    # Find contours
    curves, hierarchy = cv2.findContours(edges_morph_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
  
    curves_filtered = [np.array(curve.squeeze()) for curve in curves if len(curve) >= 5]  # filter contours with less than 5 points
    
    if debug_plot:
        curves_map = np.zeros_like(edges)
        for curve in curves:
            for p in curve:
                curves_map[p[0,1], p[0,0]] = 1
        curves_filtered_map = np.zeros_like(edges)
        for curve in curves_filtered:
            for p in curve:
                curves_filtered_map[p[1],p[0]] = 1
            
    # Filter contours with too large diameter
    # sum of all combinations of points in curve
    curves_filtered_dia = []
    for curve in curves_filtered:
        idxs = np.meshgrid(np.array(range(len(curve))), np.array(range(len(curve))))
        ps = np.array([curve[idxs[0].flatten()], curve[idxs[1].flatten()]])
        dists = np.linalg.norm(ps[0,:] - ps[1,:], axis=1)
        max_dist = np.max(dists)
        if max_dist > min(frame.shape)*.05:  # max diameter 10% of image size
            curves_filtered_dia.append(curve)
            
    if debug_plot:
        curves_filtered_dia_map = np.zeros_like(edges)
        for curve in curves_filtered_dia:
            for p in curve:
                curves_filtered_dia_map[p[1],p[0]] = 1
    
    # Fit ellipses to remaining contours
    curves_filtered_ellipse_params = []
    ellipses = []
    ellipses_unfiltered = []
    for curve in curves_filtered_dia:        
        center, size, angle = cv2.fitEllipse(curve)
        ellipses_unfiltered.append((center, size, angle))
        center_rect, size_rect, angle_rect = cv2.minAreaRect(curve)
        if min(size) > 10:  # ellipse too small?
            if center[0] >= 0 and center[0] <= edges.shape[1] and center[1] >= 0 and center[1] <= edges.shape[0]:  # center
                if size[0]/size[1] > .2:  # aspect ratio
                    if size_rect[0]/size_rect[1] > .2:
                        ellipses.append((center, size, angle))
                        curves_filtered_ellipse_params.append(curve)
                    
    if debug_plot:
        curves_filtered_ellipse_params_map = np.zeros_like(edges)
        for curve in curves_filtered_ellipse_params:
            for p in curve:
                curves_filtered_ellipse_params_map[p[1],p[0]] = 1
        
        ellipses_map = np.zeros_like(edges)
        for ellipse in ellipses:
            ellipses_map = cv2.ellipse(ellipses_map, (int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0]//2), int(ellipse[1][1]//2)), ellipse[2], 0, 360, 1, 1)
        ellipses_unfiltered_map = np.zeros_like(edges)
        for ellipse in ellipses_unfiltered:
            ellipses_unfiltered_map = cv2.ellipse(ellipses_unfiltered_map, (int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0]//2), int(ellipse[1][1]//2)), ellipse[2], 0, 360, 1, 1)
    
    ## 3.5 Conditional Segment Combination
    
    curves_cartesian = itertools.combinations_with_replacement(np.arange(0, len(curves_filtered_ellipse_params), 1), 2)
    
    def bounding_box(points):
        x_coordinates, y_coordinates = zip(*points)
        return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]
    
    combined_curves = []
    combined_ellipses = []
    combined_indices = []
    
    for combination in curves_cartesian:
        if combination[0] == combination[1]:
            continue
        
        box0 = bounding_box(curves_filtered_ellipse_params[combination[0]])
        box1 = bounding_box(curves_filtered_ellipse_params[combination[1]])
        # Check if boxes intersect
        # Don't care for now, find contrasts...
        combined_curve = np.concatenate((curves_filtered_ellipse_params[combination[0]], curves_filtered_ellipse_params[combination[1]]))
        combined_ellipse = cv2.fitEllipse(combined_curve)
        
        combined_curves.append(combined_curve)
        combined_ellipses.append(combined_ellipse)
        combined_indices.append(combination)
        
        # TODO implement Iris filter
        
    
    ## 3.4 Confidence Measure
    
    def calculate_confidence_measure(curve, ellipse, segments, combined=False):
        # Confidence measure ellipse aspect ratio
        conf_aspect_ratio = ellipse[1][0] / ellipse[1][1]
        
        # Confidence measure ellipse outline contrast
        center_int = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes_int = (int(ellipse[1][0]//2), int(ellipse[1][1]//2))
        angle_int = int(ellipse[2])
        
        ellipse_outline_contrast_list = [calculate_outline_contrast_dig(frame, center_int, p, axes_int[1]) for p in cv2.ellipse2Poly(center_int, axes_int, angle_int, 0, 360, 10)]
        conf_ellipse_outline_contrast = np.mean(ellipse_outline_contrast_list)
        
        # Confidence measure angular edge spread
        # Find curve points in segments list -> segments are complete, not just relevant points
        full_segments = []
        for segment in segments:
            if np.array([(np.array([curve[i][1], curve[i][0]]) == segment).all(1) for i in range(len(curve))]).any():  # find if any point in segment is curve point
                full_segments.append(np.array(segment))
                if not combined:
                    break
                        
        if len(full_segments) > 0:
            # calculate angular spread of segment points around ellipse center
            center = np.array((ellipse[0][1], ellipse[0][0]))
            if len(full_segments) > 1:
                all_full_segments = np.concatenate(full_segments)
            else:
                all_full_segments = full_segments[0]
            vecs = all_full_segments - center
            angles = np.arctan2(vecs[:,1], vecs[:,0])
            #angles = np.sort(angles)
            #angle_diffs = np.diff(np.concatenate((angles, angles[:1]+2*np.pi)))
            #conf_angular_edge_spread = 2*np.pi - np.max(angle_diffs)
            
            n_bins = 4   # can we go finer?
            intervals = np.linspace(-np.pi, np.pi, n_bins+1)
            # histogram of angles
            hist, _ = np.histogram(angles, bins=intervals)
            conf_angular_edge_spread = np.sum(hist > 0) / n_bins
            
            segment_length = len(all_full_segments)
            
            # calculate coverage of ellipse by segment points
            #ellipse_perimeter = np.pi * (3*(ellipse[1][0]/2 + ellipse[1][1]/2) - np.sqrt((3*ellipse[1][0]/2 + ellipse[1][1]/2)*(ellipse[1][0]/2 + 3*ellipse[1][1]/2)))
            #coverage = len(full_segment) / ellipse_perimeter
        
        else:
            print("no segment found for curve")
            conf_angular_edge_spread = 0
            segment_length = 0
        
        # Overall confidence measure
        combined_confidence_measure = np.mean([conf_aspect_ratio, conf_angular_edge_spread, conf_ellipse_outline_contrast])
        if conf_ellipse_outline_contrast < .5:
            combined_confidence_measure = 0
        
        return combined_confidence_measure, [conf_aspect_ratio, conf_ellipse_outline_contrast, conf_angular_edge_spread], segment_length
    
    if len(curves_filtered_ellipse_params) == 0:
        if debug_plot:
            fig, ax = plt.subplots(3, 4, figsize=(12, 9))
            curr_ax = ax[0,0]
            curr_ax.imshow(frame, cmap='gray')
            curr_ax.set_title('Original')
            curr_ax = ax[0,1]
            curr_ax.imshow(edges, cmap='gray')
            curr_ax.set_title('Canny Edges')
            curr_ax = ax[0,2]
            curr_ax.imshow(edges_morph_filtered, cmap='gray')
            curr_ax.set_title('Morphological Filtered')
            curr_ax = ax[0,3]
            curr_ax.imshow(curves_map, cmap='gray')
            curr_ax.set_title('Contours')
            curr_ax = ax[1,0]
            curr_ax.imshow(curves_filtered_map, cmap='gray')
            curr_ax.set_title('Filtered by point count')
            curr_ax = ax[1,1]
            curr_ax.imshow(curves_filtered_dia_map, cmap='gray')
            curr_ax.set_title('Filtered by diameter')
            curr_ax = ax[1,2]
            curr_ax.imshow(curves_filtered_ellipse_params_map, cmap='gray')
            curr_ax.set_title('Filtered by ellipse params')
            curr_ax = ax[1,3]
            curr_ax.imshow(ellipses_unfiltered_map, cmap='gray')
            curr_ax.set_title('Fitted ellipses')
            curr_ax = ax[2,0]
            curr_ax.remove()
            curr_ax = ax[2,1]
            curr_ax.remove()
            curr_ax = ax[2,2]
            curr_ax.remove()
            curr_ax = ax[2,3]
            curr_ax.remove()
            plt.tight_layout()
            plt.show()
        
        return None, 0
        
    segments = calculate_chains_for_frame(edges_morph_filtered)
    
    combined_confidences = []
    confidences = []
    for curve, ellipse in zip(curves_filtered_ellipse_params, ellipses):
        combined_confidence_measure, confidence_measures, segment_length = calculate_confidence_measure(curve, ellipse, segments, False)
        
        combined_confidences.append(combined_confidence_measure)
        confidences.append(confidence_measures)
            
        if debug_plot:
            print(f"Confidence measures: aspect ratio {confidence_measures[0]:0.2f}, angular edge spread {confidence_measures[2]:0.2f}, ellipse outline contrast {confidence_measures[1]:0.2f} -> overall {combined_confidence_measure:0.2f}, segment length {segment_length}")
    
    condcomb_curves = curves_filtered_ellipse_params
    condcomb_ellipses = ellipses
    condcomb_combined_confidences = combined_confidences
    condcomb_confidences = confidences
    
    max_conf_ellipse_outline_contrast = np.max(np.array(confidences)[:,1])
    if debug_plot:
        print(f"Maximum ellipse outline contrast: {max_conf_ellipse_outline_contrast:0.2f}")
    
    for curve, ellipse, idxs in zip(combined_curves, combined_ellipses, combined_indices):
        combined_confidence_measure, confidence_measures, segment_length = calculate_confidence_measure(curve, ellipse, segments, True)
        
        if debug_plot:
            print(f"CSC {list(idxs)} Confidence measures: aspect ratio {confidence_measures[0]:0.2f}, angular edge spread {confidence_measures[2]:0.2f}, ellipse outline contrast {confidence_measures[1]:0.2f} -> overall {combined_confidence_measure:0.2f}, segment length {segment_length}")
        
        # Check if combined curve improves outline contrasts (other metrics are likely to be improved)
        if confidence_measures[1] >= max_conf_ellipse_outline_contrast:
            condcomb_curves.append(curve)
            condcomb_ellipses.append(ellipse)
            condcomb_combined_confidences.append(combined_confidence_measure)
            condcomb_confidences.append(confidence_measures)
            
    
    if len(combined_confidences) == 0:
        return None, 0
        
    max_idx = np.argmax(combined_confidences)
    max_confidence = combined_confidences[max_idx] if len(combined_confidences) > 0 else 0
    likely_pupil_ellipse = ellipses[max_idx] if len(ellipses) > 0 else None
    
    # CV2 always returns larger height than width, which contradicts assumptions in later algorithms
    if likely_pupil_ellipse is not None:
        if likely_pupil_ellipse[1][1] > likely_pupil_ellipse[1][0]:  # should always be true with CV2
            # Swap axes -> same result, different ordering
            likely_pupil_ellipse = ((likely_pupil_ellipse[0][0], likely_pupil_ellipse[0][1]), (likely_pupil_ellipse[1][1], likely_pupil_ellipse[1][0]), (likely_pupil_ellipse[2] + 90) % 180)
    
    if debug_plot:
        ellipses_conf_map = np.zeros_like(edges)
        for confidence, ellipse in zip(combined_confidences, ellipses):
            ellipses_conf_map = cv2.ellipse(ellipses_conf_map, (int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0]//2), int(ellipse[1][1]//2)), ellipse[2], 0, 360, confidence*255, 1)
    
        ellipses_fin_map = frame.copy()
        ellipses_fin_map = cv2.cvtColor(ellipses_fin_map, cv2.COLOR_GRAY2RGB)
        ellipses_fin_map = cv2.ellipse(ellipses_fin_map, (int(ellipses[max_idx][0][0]), int(ellipses[max_idx][0][1])), (int(ellipses[max_idx][1][0]//2), int(ellipses[max_idx][1][1]//2)), ellipses[max_idx][2], 0, 360, (0, 255, 0), 1)
        
        ellipses_fin_map_all = ellipses_fin_map.copy()
        for idx, (confidence, ellipse) in enumerate(zip(combined_confidences, ellipses)):
            if idx != max_idx:
                ellipses_fin_map_all = cv2.ellipse(ellipses_fin_map_all, (int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0]//2), int(ellipse[1][1]//2)), ellipse[2], 0, 360, ((1-confidence)*255, confidence*255, 0), 1)
        
    if debug_plot:
        condcomb_ellipses_conf_map = np.zeros_like(edges)
        for confidence, ellipse in zip(combined_confidences, ellipses):
            condcomb_ellipses_conf_map = cv2.ellipse(condcomb_ellipses_conf_map, (int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0]//2), int(ellipse[1][1]//2)), ellipse[2], 0, 360, confidence*255, 1)
    
        condcomb_ellipses_fin_map = frame.copy()
        condcomb_ellipses_fin_map = cv2.cvtColor(condcomb_ellipses_fin_map, cv2.COLOR_GRAY2RGB)
        condcomb_ellipses_fin_map = cv2.ellipse(condcomb_ellipses_fin_map, (int(ellipses[max_idx][0][0]), int(ellipses[max_idx][0][1])), (int(ellipses[max_idx][1][0]//2), int(ellipses[max_idx][1][1]//2)), ellipses[max_idx][2], 0, 360, (0, 255, 0), 1)
        
        condcomb_ellipses_fin_map_all = condcomb_ellipses_fin_map.copy()
        for idx, (confidence, ellipse) in enumerate(zip(combined_confidences, ellipses)):
            if idx != max_idx:
                condcomb_ellipses_fin_map_all = cv2.ellipse(condcomb_ellipses_fin_map_all, (int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0]//2), int(ellipse[1][1]//2)), ellipse[2], 0, 360, ((1-confidence)*255, confidence*255, 0), 1)
        
    # Plot results
    if debug_plot:
        fig, ax = plt.subplots(3, 4, figsize=(12, 9))
        curr_ax = ax[0,0]
        curr_ax.imshow(frame, cmap='gray')
        curr_ax.set_title('Original')
        curr_ax = ax[0,1]
        curr_ax.imshow(edges, cmap='gray')
        curr_ax.set_title('Canny Edges')
        curr_ax = ax[0,2]
        curr_ax.imshow(edges_morph_filtered, cmap='gray')
        curr_ax.set_title('Morphological Filtered')
        curr_ax = ax[0,3]
        curr_ax.imshow(curves_map, cmap='gray')
        curr_ax.set_title('Contours')
        curr_ax = ax[1,0]
        curr_ax.imshow(curves_filtered_map, cmap='gray')
        curr_ax.set_title('Filtered by point count')
        curr_ax = ax[1,1]
        curr_ax.imshow(curves_filtered_dia_map, cmap='gray')
        curr_ax.set_title('Filtered by diameter')
        curr_ax = ax[1,2]
        curr_ax.imshow(curves_filtered_ellipse_params_map, cmap='gray')
        curr_ax.set_title('Filtered by ellipse params')
        curr_ax = ax[1,3]
        curr_ax.imshow(ellipses_unfiltered_map, cmap='gray')
        curr_ax.set_title('Fitted ellipses')
        curr_ax = ax[2,0]
        curr_ax.imshow(ellipses_conf_map, cmap='gray')
        curr_ax.set_title('Fitted with confidence')
        curr_ax = ax[2,1]
        curr_ax.imshow(ellipses_fin_map_all)
        curr_ax.set_title('Pupil fit with all candidates')
        curr_ax = ax[2,2]
        curr_ax.imshow(condcomb_ellipses_fin_map_all)
        curr_ax.set_title('Conditional seg combination')
        curr_ax = ax[2,3]
        curr_ax.imshow(ellipses_fin_map)
        curr_ax.set_title('Pupil fit')
        plt.tight_layout()
        plt.show()
    
    # TODO fit canny parameters / reimplementation of canny
    # TODO elliptical fourier series
    
    return likely_pupil_ellipse, max_confidence


def run_pd_for_sample(curr_file_name, img_folder_path='../../datasets/Openeds19/Semantic_Segmentation_Dataset/Semantic_Segmentation_Dataset/train/images', labels_folder_path='../../datasets/Openeds19/Semantic_Segmentation_Dataset/Semantic_Segmentation_Dataset/train/labels', alpha=1, beta=0, imgres=None, blur=False, canny_params=[50, 200], debug_plot=False):
    # Dataloader
    with open(f'{img_folder_path}/{curr_file_name}.png', 'rb') as f:
        frame = np.frombuffer(f.read(), np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)
        # increase brightness and contrast
        # TODO temporary fix for dark images
        if alpha is not None:
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        else:
            frame, _, _ = automatic_brightness_and_contrast(frame)
        if blur:
            frame = cv2.blur(frame, (3,3))  # Works for stereo dataset
        if imgres is not None:
            # Resize image but keep aspect ratio
            frame = cv2.resize(frame, (min(imgres, frame.shape[1] * imgres // frame.shape[0]), min(imgres, frame.shape[0] * imgres // frame.shape[1])))
    
    if debug_plot:
        plt.imshow(frame, cmap='gray')
        plt.title('Input Image')
        plt.show()
    
    likely_pupil_ellipse, max_confidence = pupil_detection(frame, canny_params, debug_plot)
    
    dimensions = [frame.shape[1], frame.shape[0]]  # axes are swapped wrt. ellipse params
    
    if likely_pupil_ellipse is None:
        if debug_plot:
            print("No pupil detected")
        return None, 0, None, dimensions, frame
    
    # Calculate loss if ground truth available
    
    def calculate_loss(gt_segmentation, pred_ellipse, debug_plot=False):
        if pred_ellipse is None:
            return 1.0
        
        gt_segmentation_mod = gt_segmentation.copy()
        gt_segmentation_mod[gt_segmentation_mod < 3] = 0  # binarize
        gt_segmentation_mod[gt_segmentation_mod == 3] = 1  # binarize
        
        def calculate_line_loss(gt_segmentation, p_center, p, length):
            dir_vec = p - p_center
            if np.linalg.norm(dir_vec) == 0:  # might happen due to too small ellipses
                return 0
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            p_in = p - dir_vec * length
            p_out = p + dir_vec * length
            inten_in = calculate_avg_intensity_along_line(gt_segmentation, p, p_in)
            inten_out = calculate_avg_intensity_along_line(gt_segmentation, p, p_out)
            score = (1 - inten_in) + inten_out  # want high inside, low outside
            
            return score
        
        center_int = (int(pred_ellipse[0][0]), int(pred_ellipse[0][1]))
        axes_int = (int(pred_ellipse[1][0]//2), int(pred_ellipse[1][1]//2))
        angle_int = int(pred_ellipse[2])
        
        ellipse_outline_contrast_list = np.mean([calculate_line_loss(gt_segmentation_mod, center_int, p, axes_int[1]) for p in cv2.ellipse2Poly(center_int, axes_int, angle_int, 0, 360, 10)])
        loss = np.mean(ellipse_outline_contrast_list)
        
        if debug_plot:
            label_map = gt_segmentation_mod.copy()
            label_map[label_map == 1] = 128
            label_map = cv2.cvtColor(label_map, cv2.COLOR_GRAY2RGB)
            label_map = cv2.ellipse(label_map, center_int, axes_int, angle_int, 0, 360, (0, 255, 0), 1)
            
            plt.imshow(label_map)
            plt.title('Ground Truth')
            plt.show()
        
        return loss
    
    if labels_folder_path is not None:
        with open(f'{labels_folder_path}/{curr_file_name}.npy' , 'rb') as f:
            label = np.lib.format.read_array(f)
            label = cv2.resize(label, (min(imgres, label.shape[1] * imgres // label.shape[0]), min(imgres, label.shape[0] * imgres // label.shape[1])))
            loss = calculate_loss(label, likely_pupil_ellipse, debug_plot)
    
        if debug_plot:
            print(f"Pupil parameters: {likely_pupil_ellipse}, Loss: {loss:0.3f}, Confidence: {max_confidence:0.3f}")
    else:
        loss = None
        if debug_plot:
            print(f"Pupil parameters: {likely_pupil_ellipse}, Confidence: {max_confidence:0.3f}")
    
    # Note that ellipse parameters are diameter, not radius!
    return likely_pupil_ellipse, max_confidence, loss, dimensions, frame
