import numpy as np
from PIL import Image
import cv2_extras as cv2x
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image as sk_image
import matplotlib.pyplot as plt
import multiprocessing
import os
import json
from micap import color_utils

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2


true_blue = 120  # in cv2 hue channel (ranges from 0 to 179)


def non_uniformity_correction(img_hsv):
    img_v = img_hsv[:, :, 2]

    # get the blue mask
    b_mask = color_utils.create_color_mask(img_hsv, colors=['blue'])

    img_v_corr = cv2x.correct_nonuniformity(img_v, mask=b_mask)
    img_hsv_corr = img_hsv.copy()
    img_hsv_corr[:, :, 2] = img_v_corr

    # repair black regions
    black_mask = color_utils.create_color_mask(img_hsv, colors=['black'])
    img_hsv_corr[black_mask > 0, 2] = img_hsv[black_mask > 0, 2]

    return img_hsv_corr


def find_color_correction_reference(hsv_imgs):
    b_h_value_counts = []
    b_h_means = []

    for i, img_hsv in enumerate(hsv_imgs):
        # get the blue mask
        # TODO: reference color should be configurable, not hard-coded
        b_mask = color_utils.create_color_mask(img_hsv, colors=['blue'])
        b_mask_img_h = cv2.bitwise_and(img_hsv[:, :, 0], img_hsv[:, :, 0], mask=b_mask)
        b_h_values = b_mask_img_h[b_mask_img_h > 0].flatten()

        b_h_value_counts.append(b_h_values.shape[0])
        b_h_means.append(np.mean(b_h_values))

    # TODO: again the reference hue should be configurable, not hard-coded
    b_h_mean_dev = np.abs(np.array(b_h_means) - true_blue)
    max_b_dev = b_h_mean_dev.max()

    b_center_devs = 1 - (b_h_mean_dev / max_b_dev)

    max_score = 0.0
    best_idx = None

    upper_count = float(max(b_h_value_counts))

    for i in range(len(hsv_imgs)):
        val_count_comp = b_h_value_counts[i] / upper_count

        score = np.mean([val_count_comp, b_center_devs[i]])

        if score > max_score:
            max_score = score
            best_idx = i

    return best_idx


def color_correction(hsv_imgs, ref_idx):
    final_corr_images_rgb = []
    ref_img_bgr = cv2.cvtColor(hsv_imgs[ref_idx], cv2.COLOR_HSV2BGR)

    for i, img_hsv in enumerate(hsv_imgs):
        if i == ref_idx:
            final_corr_images_rgb.append(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
            continue

        tar_img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        cor_img_bgr = cv2x.color_transfer(ref_img_bgr, tar_img_bgr, clip=True, preserve_paper=True)
        cor_img_rgb = cv2.cvtColor(cor_img_bgr, cv2.COLOR_BGR2RGB)
        final_corr_images_rgb.append(cor_img_rgb)

    return final_corr_images_rgb


def get_training_data_for_image_set(image_set_dir):
    # Each image set directory will have a 'regions.json' file. This regions file
    # has keys of the image file names in the image set, and the value for each image
    # is a dict of class labels, and the value of those labels is a list of
    # segmented polygon regions.
    # First, we will read in this file and get the file names for our images
    regions_file = open(os.path.join(image_set_dir, 'regions.json'))
    regions_json = json.load(regions_file)
    regions_file.close()

    # output will be a dictionary of training data, were the polygon points dict
    # is a numpy array. The keys will still be the image names
    training_data = {}

    for image_name, regions_dict in regions_json.items():
        print("Retrieving training data for %s" % image_name)
        tmp_image = Image.open(os.path.join(image_set_dir, image_name))
        tmp_image = np.asarray(tmp_image)

        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2HSV)

        training_data[image_name] = {
            'hsv_img': tmp_image,
            'regions': []
        }

        non_bg_regions = []

        for label, regions in regions_dict['regions'].items():

            for region in regions:
                points = np.array(region, dtype='int')

                training_data[image_name]['regions'].append(
                    {
                        'label': label,
                        'points': points
                    }
                )

                non_bg_regions.append(points)

        if regions_dict['full']:
            bg_contours = cv2x.generate_background_contours(
                tmp_image,
                non_bg_regions
            )

            for bg_contour in bg_contours:
                training_data[image_name]['regions'].append(
                    {
                        'label': 'background',
                        'points': bg_contour
                    }
                )

    return training_data


def process_image(img_hsv, blur_kernel_small=(15, 15), blur_kernel_large=(127, 127)):
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img_shape = (img_hsv.shape[0], img_hsv.shape[1])

    b_over_r = img_rgb[:, :, 0] < img_rgb[:, :, 2]
    b_over_g = img_rgb[:, :, 1] < img_rgb[:, :, 2]
    b_over_rg = np.bitwise_and(b_over_r, b_over_g)

    b_replace = np.max([img_rgb[:, :, 0], img_rgb[:, :, 1]], axis=0)

    b_suppress_img = img_rgb.copy()
    b_suppress_img[b_over_rg, 2] = b_replace[b_over_rg]
    b_suppress_img_hsv = cv2.cvtColor(b_suppress_img, cv2.COLOR_RGB2HSV)
    enhanced_v_img = b_suppress_img_hsv[:, :, 2]

    # diff of Gaussian blurs
    img_blur_1 = cv2.blur(enhanced_v_img, blur_kernel_small)
    img_blur_2 = cv2.blur(enhanced_v_img, blur_kernel_large)

    tmp_img_1 = img_blur_1.astype(np.int16)
    tmp_img_2 = img_blur_2.astype(np.int16)

    edge_mask = tmp_img_2 - tmp_img_1
    edge_mask[edge_mask > 0] = 0
    edge_mask[edge_mask < 0] = 255

    edge_mask = edge_mask.astype(np.uint8)

    contours = cv2x.filter_contours_by_size(edge_mask, min_size=63 * 63)

    edge_mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(edge_mask, contours, -1, 255, -1)

    # dilate just a bit
    edge_mask = cv2.dilate(edge_mask, (3, 3), iterations=blur_kernel_small[0])

    # get black mask as black can never be a "signal"
    # TODO: should 'holes' mask color be configurable? will it always be black?
    black_mask = color_utils.create_color_mask(img_hsv, colors=['black'])
    edge_mask = np.bitwise_and(edge_mask, ~black_mask)

    return b_suppress_img_hsv, edge_mask, black_mask


def generate_color_contours(
        img_hsv,
        blur_kernel=(9, 9),
        mask=None,
        min_size=None,
        max_size=None,
        colors=('green', 'cyan', 'red', 'violet', 'yellow')
):
    img_shape = (img_hsv.shape[0], img_hsv.shape[1])

    tmp_color_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    tmp_color_img = cv2.blur(tmp_color_img, blur_kernel)
    tmp_color_img = cv2.cvtColor(tmp_color_img, cv2.COLOR_RGB2HSV)

    color_mask = color_utils.create_color_mask(tmp_color_img, colors)

    color_mask = cv2.bitwise_and(
        color_mask,
        color_mask,
        mask=mask
    )

    # Next, clean up any "noise", say, ~ 5 x 5
    contours = cv2x.filter_contours_by_size(color_mask, min_size=5 * 5)

    color_mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(color_mask, contours, -1, 255, -1)

    # do a couple dilation iterations to smooth some outside edges & connect any
    # adjacent cells each other
    color_mask = cv2.dilate(color_mask, cv2x.cross_strel, iterations=2)
    color_mask = cv2.erode(color_mask, cv2x.cross_strel, iterations=2)

    # filter remaining contours by size
    contours = cv2x.filter_contours_by_size(
        color_mask,
        min_size=min_size,
        max_size=max_size
    )

    print("%d color candidates found, kernel:" % len(contours), blur_kernel)

    return contours


def generate_saturation_contours(
        img_s,
        blur_kernel,
        mask=None,
        min_size=None,
        max_size=None
):
    img_s_blur = cv2.GaussianBlur(img_s, blur_kernel, 0, 0)

    med = np.median(img_s_blur[img_s_blur > 0])

    img_s_blur = cv2.bitwise_and(
        img_s_blur,
        img_s_blur,
        mask=mask
    )
    mode_s = cv2.inRange(img_s_blur, med, 255)
    mode_s = cv2.erode(mode_s, cv2x.block_strel, iterations=2)

    contours = cv2x.filter_contours_by_size(
        mode_s,
        min_size=min_size,
        max_size=max_size
    )

    print("%d saturation candidates found, kernel:" % len(contours), blur_kernel)

    return contours


def dilate_contours_by_signal_mask(contours, signal_mask):
    new_contour_results = []
    pool = multiprocessing.Pool()

    for c in contours:
        new_contour_results.append(
            pool.apply_async(
                cv2x.find_border_by_mask,
                (signal_mask, c)
            )
        )
    pool.close()

    good_contours = []

    for res in new_contour_results:
        new_mask, signal, orig = res.get()

        if signal > 0.7:
            tmp_contours, _ = cv2.findContours(
                new_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            good_contours.append(tmp_contours[0])
        else:
            pass  # ignore contour

    return good_contours


def update_edge_mask(edge_mask, candidate_mask=None, holes_mask=None, filter_min_size=63 * 63):
    # update the signal mask by removing the previous candidates
    if candidate_mask is not None:
        edge_mask = cv2.bitwise_and(
            edge_mask,
            edge_mask,
            mask=~candidate_mask
        )

    edge_contours = cv2x.filter_contours_by_size(
        edge_mask,
        min_size=filter_min_size
    )
    edge_mask = np.zeros(edge_mask.shape, dtype=np.uint8)
    cv2.drawContours(edge_mask, edge_contours, -1, 255, -1)

    if holes_mask is not None:
        # "un-fill" the holes
        edge_mask = np.bitwise_and(edge_mask, ~holes_mask)

    return edge_mask


def create_color_soft_mask(hue_center, hsv_img, clip=1.0):
    value_multiplier = np.cos(2 * np.pi * ((hsv_img[:, :, 0] - hue_center) / 180)) / 2 + .5
    value_multiplier[value_multiplier < 0] = 0
    value_multiplier[value_multiplier > clip] = clip
    soft_mask = hsv_img[:, :, 2] * value_multiplier

    return soft_mask


def split_multi_cell(signal_img, multi_cell_mask, max_cell_area, plot=False):
    contour_tree, hierarchy = cv2.findContours(
        multi_cell_mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    single_cell_contours = []
    sub_multi_contours_idx = []

    for c_idx, sub_c in enumerate(contour_tree):
        h = hierarchy[0][c_idx]
        if h[3] != -1:
            # it's a child contour, ignore it
            continue

        sub_c_area = cv2.contourArea(sub_c)
        if sub_c_area < max_cell_area * .33:
            # too small, some kind of fragment
            continue
        elif sub_c_area > max_cell_area * 1.1:
            # too big for a single cell, try to split it
            sub_multi_contours_idx.append(c_idx)
        else:
            # just right, probably a cell so save it
            single_cell_contours.append(sub_c)

            if plot:
                sc_mask = np.zeros(signal_img.shape, dtype=np.uint8)
                cv2.drawContours(
                    sc_mask,
                    [sub_c],
                    -1,
                    255,
                    cv2.FILLED
                )
                plt.figure(figsize=(16, 16))
                plt.imshow(sc_mask)
                plt.axis('off')
                plt.show()

    for c_idx in sub_multi_contours_idx:

        mc_mask = np.zeros(signal_img.shape, dtype=np.uint8)
        cv2.drawContours(
            mc_mask,
            contour_tree,
            c_idx,
            255,
            cv2.FILLED,
            hierarchy=hierarchy
        )

        # Convert the image into a graph with the value of the gradient on the
        # edges.
        region_graph = sk_image.img_to_graph(
            signal_img,
            mask=mc_mask.astype(np.bool)
        )

        # Take a decreasing function of the gradient: we take it weakly
        # dependent from the gradient the segmentation is close to a voronoi
        region_graph.data = np.exp(-region_graph.data / region_graph.data.std())

        n_clusters = 2

        labels = spectral_clustering(
            region_graph,
            n_clusters=n_clusters,
            eigen_solver='arpack',
            n_init=10
        )

        label_im = np.full(mc_mask.shape, -1.)
        label_im[mc_mask.astype(np.bool)] = labels

        if plot:
            plt.figure(figsize=(16, 16))
            plt.imshow(label_im)
            plt.axis('off')
            plt.show()

        for label in range(n_clusters):
            new_mask = label_im == label

            single_cell_contours.extend(
                split_multi_cell(signal_img, new_mask, max_cell_area, plot=plot)
            )

    return single_cell_contours


def numpy_json_converter(o):
    if isinstance(o, np.ndarray):
        return o.tolist()


def process_structures_into_cells(
        img_hsv,
        save_dir,
        structure_contours,
        cell_color_list,
        max_cell_area,
        plot=False
):
    structure_contours_with_cells = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img_shape = (
        img_hsv.shape[0],
        img_hsv.shape[1]
    )

    # final contour mask for viewing residual areas
    final_contour_mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(final_contour_mask, structure_contours, -1, 255, -1)

    seg_image = img_rgb.copy()
    cv2.drawContours(seg_image, structure_contours, -1, (0, 255, 0), 3)
    cv2x.save_image(save_dir, '01_structures.tif', seg_image)

    likely_multi_cells = []

    for c in structure_contours:
        area = cv2.contourArea(c)

        if area > max_cell_area:
            likely_multi_cells.append(c)
        else:
            structure_contours_with_cells.append(
                {
                    'region': c,
                    'cells': [c]
                }
            )

    soft_masks = []
    for color in cell_color_list:
        hsv_bounds = color_utils.HSV_RANGES[color]

        for hsv_bound in hsv_bounds:
            h_lower = hsv_bound['lower'][0]
            h_upper = hsv_bound['upper'][0]
            soft_mask = create_color_soft_mask(
                np.mean([h_lower, h_upper]),
                img_hsv,
                clip=1.0
            )
            soft_masks.append(soft_mask)

    signal_soft_mask = np.mean(soft_masks, axis=0)

    ret, non_dark_thresh = cv2.threshold(signal_soft_mask, 95, 255, cv2.THRESH_BINARY)

    for i, mc in enumerate(likely_multi_cells):
        print("multi-cell %d" % i)
        min_x, min_y, w, h = cv2.boundingRect(mc)
        max_x, max_y = min_x + w, min_y + h
        mc_x = mc - [min_x, min_y]

        multi_cell_mask_x = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(multi_cell_mask_x, [mc_x], -1, 1, -1)
        multi_cell_mask_x = multi_cell_mask_x.astype(np.bool)
        non_dark_thresh_x = non_dark_thresh[min_y:max_y, min_x:max_x]
        final_multi_cell_mask_x = cv2.bitwise_and(
            multi_cell_mask_x.astype(np.uint8),
            non_dark_thresh_x.astype(np.uint8)
        )
        final_multi_cell_mask_x = cv2.erode(
            final_multi_cell_mask_x,
            cv2x.cross_strel,
            iterations=1
        )
        final_multi_cell_mask_x = cv2.dilate(
            final_multi_cell_mask_x,
            cv2x.cross_strel,
            iterations=1
        )

        signal_img_x = signal_soft_mask[min_y:max_y, min_x:max_x].astype('float')

        if plot:
            plot_img_mc = seg_image[min_y:max_y, min_x:max_x].copy()
            cv2.drawContours(
                plot_img_mc,
                [mc_x],
                -1,
                (0, 255, 0),
                3
            )
            plt.figure(figsize=(16, 16))
            plt.imshow(plot_img_mc)
            plt.axis('off')
            plt.show()

        new_contours = split_multi_cell(
            signal_img_x,
            final_multi_cell_mask_x,
            max_cell_area,
            plot=plot
        )

        final_sc_contours = []
        non_trans_smooth_contours = []

        for new_contour in new_contours:
            cell_mask = np.zeros(signal_img_x.shape, dtype=np.uint8)
            cv2.drawContours(
                cell_mask,
                [new_contour],
                -1,
                255,
                cv2.FILLED
            )
            cell_mask = cv2.morphologyEx(
                cell_mask,
                cv2.MORPH_OPEN,
                cv2x.cross_strel,
                iterations=4
            )
            cell_mask = cv2.dilate(
                cell_mask,
                cv2x.circle_strel,
                iterations=1
            )
            morphed_contours, _ = cv2.findContours(
                cell_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for c in morphed_contours:
                peri = cv2.arcLength(c, True)
                smooth_c = cv2.approxPolyDP(c, 0.015 * peri, True)
                non_trans_smooth_contours.append(smooth_c)
                final_sc_contours.append(smooth_c + [min_x, min_y])

        structure_contours_with_cells.append(
            {
                'region': mc,
                'cells': final_sc_contours
            }
        )

        if plot:
            plot_img_mc = seg_image[min_y:max_y, min_x:max_x].copy()
            cv2.drawContours(
                plot_img_mc,
                non_trans_smooth_contours,
                -1,
                (0, 255, 0),
                1
            )
            cv2.drawContours(
                plot_img_mc,
                [mc_x],
                -1,
                (0, 255, 0),
                3
            )
            plt.figure(figsize=(16, 16))
            plt.imshow(plot_img_mc)
            plt.axis('off')
            plt.show()

    seg_image = img_rgb.copy()
    for structure in structure_contours_with_cells:
        if len(structure['cells']) > 0:
            cv2.drawContours(seg_image, structure['cells'], -1, (128, 255, 255), 1)

    cv2x.save_image(save_dir, '02_single_cells.tif', seg_image)

    return structure_contours_with_cells
