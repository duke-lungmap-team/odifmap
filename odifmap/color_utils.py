import numpy as np
# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2
import pandas as pd
from scipy.spatial.distance import pdist
import cv2_extras as cv2x

np.seterr(all='warn')


HSV_RANGES = {
    # red is a major color
    'red': [
        {
            'lower': np.array([0, 39, 64]),
            'upper': np.array([19, 255, 255])
        },
        {
            'lower': np.array([160, 39, 64]),
            'upper': np.array([179, 255, 255])
        }
    ],
    # yellow is a minor color
    'yellow': [
        {
            'lower': np.array([20, 39, 64]),
            'upper': np.array([39, 255, 255])
        }
    ],
    # green is a major color
    'green': [
        {
            'lower': np.array([40, 39, 64]),
            'upper': np.array([79, 255, 255])
        }
    ],
    # cyan is a minor color
    'cyan': [
        {
            'lower': np.array([80, 39, 64]),
            'upper': np.array([99, 255, 255])
        }
    ],
    # blue is a major color
    'blue': [
        {
            'lower': np.array([100, 39, 64]),
            'upper': np.array([139, 255, 255])
        }
    ],
    # violet is a minor color
    'violet': [
        {
            'lower': np.array([140, 39, 64]),
            'upper': np.array([159, 255, 255])
        }
    ],
    # next are the monochrome ranges
    # black is all H & S values, but only the lower 10% of V
    'black': [
        {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([179, 255, 63])
        }
    ],
    # gray is all H values, lower 15% of S, & between 11-89% of V
    'gray': [
        {
            'lower': np.array([0, 0, 64]),
            'upper': np.array([179, 38, 228])
        }
    ],
    # white is all H values, lower 15% of S, & upper 10% of V
    'white': [
        {
            'lower': np.array([0, 0, 229]),
            'upper': np.array([179, 38, 255])
        }
    ]
}


def create_color_mask(hsv_img, colors):
    """
    Creates a binary mask from HSV image using given colors.
    """
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )

    return mask


def get_color_profile(hsv_img, mask=None):
    """
    Finds color profile as pixel counts for color ranges in HSV_RANGES

    Args:
        hsv_img: HSV pixel data (3-D NumPy array)
        mask: optional mask to apply to hsv_img

    Returns:
        Text string for dominant color range (from HSV_RANGES keys)

    Raises:
        tbd
    """
    h, s, v = cv2x.get_flat_hsv_channels(hsv_img, mask)

    color_profile = {}

    for color, color_ranges in HSV_RANGES.items():
        color_profile[color] = 0

        for color_range in color_ranges:
            pixel_count = np.sum(
                np.logical_and(
                    h >= color_range['lower'][0],
                    h <= color_range['upper'][0]
                ) &
                np.logical_and(
                    s >= color_range['lower'][1],
                    s <= color_range['upper'][1]
                ) &
                np.logical_and(
                    v >= color_range['lower'][2],
                    v <= color_range['upper'][2]
                )
            )

            color_profile[color] += pixel_count

    return color_profile


def get_target_features(hsv_img, mask=None):
    h, s, v = cv2x.get_flat_hsv_channels(hsv_img, mask)
    s = s / 255.0
    v = v / 255.0
    color_features = get_color_features(hsv_img, mask=mask)

    feature_names = []
    values = []

    # add region features first
    feature_names.append('region_area')
    values.append(len(h))

    feature_names.append('region_saturation_mean')
    values.append(np.mean(s))

    feature_names.append('region_saturation_variance')
    values.append(np.var(s))

    feature_names.append('region_value_mean')
    values.append(np.mean(v))

    feature_names.append('region_value_variance')
    values.append(np.var(v))

    if mask is not None:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # get smallest bounding rectangle (rotated)
        box = cv2.minAreaRect(contours[0])
        cnt_w, cnt_h = box[1]

        region_eccentricity = cnt_w / cnt_h
        if region_eccentricity > 1:
            region_eccentricity = 1.0 / region_eccentricity

        peri = cv2.arcLength(contours[0], True)

        # calculate circularity as (4 * pi * area) / perimeter ^ 2
        region_circularity = (4 * np.pi * len(h)) / float(peri) ** 2

        # calculate convexity as convex hull perimeter / contour perimeter
        hull = cv2.convexHull(contours[0])
        region_convexity = cv2.arcLength(hull, True) / peri

        feature_names.append('region_eccentricity')
        values.append(region_eccentricity)

        feature_names.append('region_circularity')
        values.append(region_circularity)

        feature_names.append('region_convexity')
        values.append(region_convexity)

    for color, features in color_features.items():
        for feature, value in sorted(features.items()):
            feature_str = '%s (%s)' % (feature, color)
            feature_names.append(feature_str)
            values.append(value)

    target_features = pd.Series(values, index=feature_names)

    return target_features.sort_index()


def get_color_features(hsv_img, mask=None):
    """
    Takes an hsv image and returns a custom set of features useful for machine learning
    :param hsv_img: np.array with color scheme hsv
    :param mask: np.array list that contains the line segments of the thing being described
    :return: dictionary of features for machine learning
    """
    c_prof = get_color_profile(hsv_img, mask)

    if mask is not None:
        tot_px = np.sum(mask > 0)
    else:
        tot_px = hsv_img.shape[0] * hsv_img.shape[1]

    # corner to corner distance used to normalize sub-contour distance metrics
    diag_distance = cv2x.calculate_distance(0, 0, hsv_img.shape[0], hsv_img.shape[1])

    color_features = {}

    for color in HSV_RANGES.keys():
        color_percent = float(c_prof[color]) / tot_px

        # create color mask & apply it
        color_mask = create_color_mask(hsv_img, [color])

        # apply user specified mask
        if mask is not None:
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)

        ret, thresh = cv2.threshold(color_mask, 1, 255, cv2.THRESH_BINARY)

        # To properly calculate the metrics, any "complex" contours with
        # holes (i.e., those with a contour hierarchy) need to be reconstructed
        # with their hierarchy. The OpenCV hierarchy scheme is a 4 column
        # NumPy array with the following convention:
        #
        #   [Next, Previous, First_Child, Parent]
        #
        #   Next: Index of next contour at the same hierarchical level
        #   Previous: Index of previous contour at the same hierarchical level
        #   First_Child: Index of the parent's first child contour
        #   Parent: Index of the parent contour
        #
        # If any of these cases do not apply then the value -1 is used. The
        # following pseudo-code covers the different cases:
        #
        #   If 'Parent' == -1:
        #     the index is a root parent contour
        #
        #     If 'First_Child' == -1:
        #       the index has no children
        #     If 'Next' == -1:
        #       there are no more root-level parent contours left
        #
        #   Else If 'Parent' > -1:
        #     the index is a child contour w/ 'Parent' value as it's parent
        #
        #     If 'Next' & 'First_Child' == -1:
        #       the child has no further siblings or children, it's a "leaf"
        #
        # This can all get quite complex, with grand-children, and further
        # nesting. Any root parent is an external contour, 1st level children
        # are then the boundary of inner holes, the root parent's grandchildren
        # would be the outer boundary of a nested contour, etc.
        #
        # However, we only want to consider connected contours, meaning those
        # external boundaries and their direct boundaries for any holes. Any
        # grandchildren we want to consider as being at the root level. For
        # this, OpenCV has a retrieval method 'RETR_CCOMP', where only these
        # 2 levels are used
        contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cent_list = []
        area_list = []
        peri_list = []
        har_list = []

        if len(contours) == 0:
            color_features[color] = {
                'percent': color_percent,
                'contour_count': len(cent_list),
                'distance_mean': 0.0,
                'distance_variance': 0.0,
                'area_mean': 0.0,
                'area_variance': 0.0,
                'perimeter_mean': 0.0,
                'perimeter_variance': 0.0,
                'har_mean': 0.0,
                'har_variance': 0.0,
                'largest_contour_area': 0.0,
                'largest_contour_saturation_mean': 0.0,
                'largest_contour_saturation_variance': 0.0,
                'largest_contour_value_mean': 0.0,
                'largest_contour_value_variance': 0.0,
                'largest_contour_eccentricity': 0.0,
                'largest_contour_circularity': 0.0,
                'largest_contour_convexity': 0.0,
                'largest_contour_har': 0.0
            }
            continue

        # If contour count is > 0, then we have hierarchies,
        # get all the parent contour indices
        parent_contour_indices = np.where(hierarchy[:, :, 3] == -1)[1]

        largest_contour_area = 0.0
        largest_contour_true_area = 0.0
        largest_contour_peri = 0.0
        largest_contour_har = 0.0
        largest_contour = None
        largest_contour_idx = None

        for c_idx in parent_contour_indices:
            c_mask = np.zeros(hsv_img.shape[0:2], dtype=np.uint8)
            cv2.drawContours(c_mask, contours, c_idx, 255, -1, hierarchy=hierarchy)

            true_area = np.sum(c_mask > 0)

            # contour may be a single point or a line, which has no area
            # we'll also ignore "noise" of anything 4 pixels or less
            # also if the contour is only a point or line, it has no area
            # ignore 'noise' of any sub-contours that are <0.1% of total area
            if true_area <= 8 or len(contours[c_idx]) <= 3:
                continue

            area = true_area / float(tot_px)
            peri = cv2.arcLength(contours[c_idx], True)
            try:
                m = cv2.moments(contours[c_idx])
                centroid_x = m['m10'] / m['m00']
                centroid_y = m['m01'] / m['m00']
            except ZeroDivisionError:
                centroid_x = contours[c_idx][:, :, 0].mean()
                centroid_y = contours[c_idx][:, :, 1].mean()

            # re-draw contour without holes
            cv2.drawContours(c_mask, contours, c_idx, 255, -1)
            filled_area = np.sum(c_mask > 0)
            hole_area_ratio = (filled_area - true_area) / float(filled_area)

            if area > largest_contour_area:
                largest_contour_area = area
                largest_contour_true_area = true_area
                largest_contour_peri = peri
                largest_contour_har = hole_area_ratio
                largest_contour = contours[c_idx]
                largest_contour_idx = c_idx

            cent_list.append((centroid_x, centroid_y))
            area_list.append(area)
            peri_list.append(peri)
            har_list.append(hole_area_ratio)

        if len(cent_list) <= 1:
            pair_dist = [0.0]
        else:
            pair_dist = pdist(np.array(cent_list)) / diag_distance

        dist_mean = np.mean(pair_dist)
        dist_var = np.var(pair_dist)

        if len(area_list) == 0:
            area_mean = 0.0
            area_var = 0.0
        else:
            area_mean = np.mean(area_list)
            area_var = np.var(area_list)

        if len(peri_list) == 0:
            peri_mean = 0.0
            peri_var = 0.0
        else:
            peri_mean = np.mean(peri_list)
            peri_var = np.var(peri_list)

        if len(har_list) == 0:
            har_mean = 0.0
            har_var = 0.0
        else:
            har_mean = np.mean(har_list)
            har_var = np.var(har_list)

        largest_contour_eccentricity = 0.0
        largest_contour_circularity = 0.0
        largest_contour_convexity = 0.0
        largest_contour_sat_mean = 0.0
        largest_contour_sat_var = 0.0
        largest_contour_val_mean = 0.0
        largest_contour_val_var = 0.0

        if largest_contour_true_area >= 0.0 and largest_contour is not None:
            lc_mask = np.zeros(hsv_img.shape[0:2], dtype=np.uint8)
            cv2.drawContours(
                lc_mask,
                contours,
                largest_contour_idx,
                255,
                cv2.FILLED,
                hierarchy=hierarchy
            )
            lc_h, lc_s, lc_v = cv2x.get_flat_hsv_channels(hsv_img, mask)
            lc_s = lc_s / 255.0
            lc_v = lc_v / 255.0

            largest_contour_sat_mean = np.mean(lc_s)
            largest_contour_sat_var = np.var(lc_s)
            largest_contour_val_mean = np.mean(lc_v)
            largest_contour_val_var = np.mean(lc_v)

            # get smallest bounding rectangle (rotated)
            box = cv2.minAreaRect(largest_contour)
            cnt_w, cnt_h = box[1]

            largest_contour_eccentricity = cnt_w / cnt_h
            if largest_contour_eccentricity > 1:
                largest_contour_eccentricity = 1.0 / largest_contour_eccentricity

            # calculate circularity as (4 * pi * area) / perimeter ^ 2
            largest_contour_circularity = (4 * np.pi * largest_contour_true_area) / float(largest_contour_peri)**2

            # calculate convexity as convex hull perimeter / contour perimeter
            hull = cv2.convexHull(largest_contour)
            largest_contour_convexity = cv2.arcLength(hull, True) / largest_contour_peri

        color_features[color] = {
            'percent': color_percent,
            'contour_count': len(cent_list),
            'distance_mean': dist_mean,
            'distance_variance': dist_var,
            'area_mean': area_mean,
            'area_variance': area_var,
            'perimeter_mean': peri_mean,
            'perimeter_variance': peri_var,
            'har_mean': har_mean,
            'har_variance': har_var,
            'largest_contour_area': largest_contour_area,
            'largest_contour_saturation_mean': largest_contour_sat_mean,
            'largest_contour_saturation_variance': largest_contour_sat_var,
            'largest_contour_value_mean': largest_contour_val_mean,
            'largest_contour_value_variance': largest_contour_val_var,
            'largest_contour_eccentricity': largest_contour_eccentricity,
            'largest_contour_circularity': largest_contour_circularity,
            'largest_contour_convexity': largest_contour_convexity,
            'largest_contour_har': largest_contour_har
        }

    return color_features


def generate_features(
        hsv_img_as_numpy,
        polygon_points,
        label=None,
        outer_margin_distance=None,
        inset_center_distance=None,
        region_file_path=None
):
    """
    Given an hsv image represented as a numpy array, polygon points which represent a
    target entity, and a label, this function will return a set of important features about
    the entity outlined in the polygons (plus some area blackened to generate a rectangle).
    :param hsv_img_as_numpy: numpy.array
    :param polygon_points: numpy.array
    :param label: str: indicating what the thing is
    :param outer_margin_distance: int: Pixel distance from region's outer margin
        from which extra features will be generated. Note: these features will
        be added to the feature set, the margin will not be included for the region's
        base features.
    :param inset_center_distance: int: Pixel distance inside region's outer
    margin from which extra features will be generated. Note: these features will
    be added to the feature set.
    :param region_file_path: str: optional file path to save the cropped sub-region as PNG
    :return: a dictionary containing features and a label key
    """
    polygon_points = polygon_points.copy()
    mask = np.zeros(hsv_img_as_numpy.shape[0:2], dtype=np.uint8)
    cv2.drawContours(mask, [polygon_points], 0, 255, cv2.FILLED)

    if inset_center_distance == 'auto':
        # calculate minimum contour dimension
        box = cv2.minAreaRect(polygon_points)
        cnt_w, cnt_h = box[1]

        if cnt_w <= cnt_h:
            min_dim = cnt_w
        else:
            min_dim = cnt_h

        inset_center_distance = int(min_dim / 6)

    if inset_center_distance is not None:
        center_mask = cv2.erode(
            mask,
            cv2x.cross_strel,
            iterations=inset_center_distance
        )

        if np.sum(center_mask > 0) <= 9:
            # the orig region was likely very spindly and eroded to nothing
            # we'll say the "center" for these is equal to the orig region
            center_mask = mask.copy()

        center_contours, _ = cv2.findContours(
            center_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # eroded contour may be broken into pieces, find largest piece
        largest_center_area = 0
        center_contour = center_contours[0]
        for c_idx, c in enumerate(center_contours):
            c_mask = np.zeros(mask.shape[0:2], dtype=np.uint8)
            cv2.drawContours(c_mask, center_contours, c_idx, 255, -1)

            area = np.sum(c_mask > 0)

            if area > largest_center_area:
                largest_center_area = area
                center_contour = center_contours[c_idx]
    else:
        center_contour = None

    if outer_margin_distance is not None:
        dilate_mask = cv2.dilate(
            mask,
            np.ones((3, 3)),
            iterations=outer_margin_distance
        )
        contours, _ = cv2.findContours(
            dilate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # find bounding coordinates of dilated mask
        x1, y1, x2, y2 = cv2x.get_bounding_rect(contours[0])

        # crop given image
        crop_img = cv2x.crop_image(hsv_img_as_numpy, x1, y1, x2, y2)

        # translate the dilated contour to cropped dimensions
        cropped_dilated_contour = cv2x.translate_contour(contours[0], x1, y1)

        # translate the orig contour to cropped dimensions
        cropped_contour = cv2x.translate_contour(polygon_points, x1, y1)

        # draw the cropped contour
        crop_mask = np.zeros(crop_img.shape[0:2], dtype=np.uint8)
        cv2.drawContours(crop_mask, [cropped_contour], 0, 255, cv2.FILLED)

        # draw the cropped dilated contour
        crop_dilate_mask = np.zeros(crop_img.shape[0:2], dtype=np.uint8)
        cv2.drawContours(
            crop_dilate_mask,
            [cropped_dilated_contour],
            0,
            255,
            cv2.FILLED
        )

        # remove the orig contour area from the dilated contour to get
        # just the margin mask
        margin_mask = cv2.bitwise_and(
            crop_dilate_mask,
            crop_dilate_mask,
            mask=~crop_mask
        )
    else:
        # crop to given mask (not dilated)
        x1, y1, x2, y2 = cv2x.get_bounding_rect(polygon_points)

        # crop given image
        crop_img = cv2x.crop_image(hsv_img_as_numpy, x1, y1, x2, y2)

        # translate the contour to cropped dimensions
        cropped_contour = cv2x.translate_contour(polygon_points, x1, y1)

        # draw the cropped contour
        crop_mask = np.zeros(crop_img.shape[0:2], dtype=np.uint8)
        cv2.drawContours(crop_mask, [cropped_contour], 0, 255, cv2.FILLED)

        margin_mask = None

    target_features = get_target_features(crop_img, mask=crop_mask)

    if inset_center_distance is not None:
        # translate the center contour to cropped dimensions
        cropped_center_contour = cv2x.translate_contour(center_contour, x1, y1)

        # draw cropped center contour
        crop_center_mask = np.zeros(crop_img.shape[0:2], dtype=np.uint8)
        cv2.drawContours(
            crop_center_mask,
            [cropped_center_contour],
            0,
            255,
            cv2.FILLED
        )

        center_features = get_target_features(crop_img, mask=crop_center_mask)
        center_features = center_features.add_prefix('center_')
        target_features = pd.concat([target_features, center_features])

    if outer_margin_distance is not None:
        margin_features = get_target_features(
            crop_img,
            mask=margin_mask
        )
        margin_features = margin_features.add_prefix('margin_')

        target_features = pd.concat([target_features, margin_features])

    if region_file_path is not None:
        crop_mask_img = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)
        cv2.imwrite(region_file_path, cv2.cvtColor(crop_mask_img, cv2.COLOR_HSV2BGR))

    results = dict(target_features.to_dict())
    results['label'] = label

    return results
