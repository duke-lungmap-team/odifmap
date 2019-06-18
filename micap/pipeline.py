import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
import cv2_extras as cv2x
from micap import utils, color_utils

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2

param = {
    'verbosity': 0,
    'max_depth': 4,
    'eta': 0.1,
    'min_child_weight': 2,
    'colsample_bytree': 0.25,
    'colsample_bylevel': 0.25,
    'objective': 'multi:softprob',
    'subsample': 0.6,
    'eval_metric': 'mlogloss'
}
num_round = 40000


def process_training_data(training_data):
    # process the training data to get custom metrics
    training_data_processed = []

    for img_name, img_data in training_data.items():
        print("Generating feature metrics for %s" % img_name)
        for region in img_data['regions']:
            training_data_processed.append(
                color_utils.generate_features(
                    hsv_img_as_numpy=img_data['hsv_img'],
                    polygon_points=region['points'],
                    label=region['label']
                )
            )

    training_data_processed = pd.DataFrame(training_data_processed)

    return training_data_processed


def fit(training_data_processed):
    coded_labels = pd.Categorical(training_data_processed['label'])
    categories = coded_labels.categories
    param['num_class'] = len(categories)

    x_train, x_eval, y_train, y_eval = train_test_split(
        training_data_processed.drop('label', axis=1),
        coded_labels.codes,
        test_size=0.33,
        stratify=coded_labels.codes,
        random_state=123
    )
    d_train = xgb.DMatrix(
        x_train,
        label=y_train
    )
    d_eval = xgb.DMatrix(
        x_eval,
        label=y_eval
    )

    watchlist = [(d_train, 'train'), (d_eval, 'eval')]

    xgb_model = xgb.train(
        param,
        d_train,
        num_round,
        evals=watchlist,
        verbose_eval=2000,
        early_stopping_rounds=4000
    )

    return xgb_model, categories


def generate_structure_candidates(
        img_hsv,
        seg_config,
        filter_min_size=None,
        dog_factor=8,
        process_residual=True,
        predict_model=None,
        categories=None,
        plot=False,
        progress_callback=None,
        use_signal_mask=True
):
    img_s = img_hsv[:, :, 1]
    img_shape = (
        img_hsv.shape[0],
        img_hsv.shape[1]
    )

    progress = 0
    process_step_count = 1
    if progress_callback is not None:
        # determine number of steps as a rough estimate of progress
        # calculated as the number of seg stages plus 1 for pre-processing
        process_step_count = 1 + len(seg_config)

    print("Pre-processing test image...")
    # determine kernel sizes for DoG
    if filter_min_size is not None:
        blur_kernel_small = int(np.sqrt(filter_min_size) / np.pi)
        if blur_kernel_small % 2 == 0:
            blur_kernel_small = blur_kernel_small - 1

        blur_kernel_large = dog_factor * blur_kernel_small
        if blur_kernel_large % 2 == 0:
            blur_kernel_large = blur_kernel_large - 1

        blur_kernel_small = (blur_kernel_small, blur_kernel_small)
        blur_kernel_large = (blur_kernel_large, blur_kernel_large)
    else:
        blur_kernel_small = (15, 15)
        blur_kernel_large = (127, 127)

    b_suppress_img_hsv, edge_mask, holes_mask = utils.process_image(
        img_hsv,
        blur_kernel_small=blur_kernel_small,
        blur_kernel_large=blur_kernel_large
    )
    if progress_callback is not None:
        progress += 1
        progress_callback(progress / float(process_step_count))

    edge_mask = utils.update_edge_mask(
        edge_mask,
        candidate_mask=None,
        holes_mask=holes_mask
    )
    if plot:
        plt.figure(figsize=(16, 16))
        plt.imshow(edge_mask)
        plt.axis('off')
        plt.show()

    candidate_mask = None
    all_contours = []

    for seg in seg_config:
        if seg['type'] == 'color':
            contours = utils.generate_color_contours(
                b_suppress_img_hsv,
                blur_kernel=seg['args']['blur_kernel'],
                min_size=seg['args']['min_size'],
                max_size=seg['args']['max_size'],
                mask=~candidate_mask if candidate_mask is not None else None,
                colors=seg['args']['colors']
            )
        elif seg['type'] == 'saturation':
            # TODO: should img_s be changed to the sat channel of the
            # b_suppressed img? does it make a difference, are they the same?
            contours = utils.generate_saturation_contours(
                img_s,
                blur_kernel=seg['args']['blur_kernel'],
                min_size=seg['args']['min_size'],
                max_size=seg['args']['max_size'],
                mask=~candidate_mask if candidate_mask is not None else None,
            )
        else:
            raise ValueError("Invalid config, use 'color' or 'saturation' for 'type'")
        if use_signal_mask:
            contours = utils.dilate_contours_by_signal_mask(
                contours,
                edge_mask
            )
            print("\t%d contours fit signal mask" % len(contours))
        contours = cv2x.smooth_contours(contours)

        if progress_callback is not None:
            progress += 1
            progress_callback(progress / float(process_step_count))

        if len(contours) == 0:
            continue

        if predict_model is not None:
            test_data_processed = process_test_data(img_hsv, contours)
            pred_results = predict(
                test_data_processed,
                predict_model,
                categories
            )

            tmp_contours = []
            for res in pred_results:
                if res['label'] == 'background':
                    continue

                tmp_contours.append(contours[res['contour_index']])

            print(
                "\t%d contours classified as background, ignoring..." %
                (len(contours) - len(tmp_contours))
            )
            contours = tmp_contours

        all_contours.extend(contours)

        # update intermediate candidate mask
        tmp_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(tmp_mask, contours, -1, 255, -1)
        candidate_mask = tmp_mask
        edge_mask = utils.update_edge_mask(
            edge_mask,
            candidate_mask=candidate_mask,
            holes_mask=holes_mask,
            filter_min_size=filter_min_size
        )
        if plot:
            plt.figure(figsize=(16, 16))
            plt.imshow(edge_mask)
            plt.axis('off')
            plt.show()

            new_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            tmp_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(tmp_mask, all_contours, -1, 255, -1)
            new_img = cv2.bitwise_and(new_img, new_img, mask=tmp_mask)

            cv2.drawContours(new_img, all_contours, -1, (0, 255, 0), 5)
            plt.figure(figsize=(16, 16))
            plt.imshow(new_img)
            plt.axis('off')
            plt.show()

    remaining_edge_contours, _ = cv2.findContours(
        edge_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if process_residual:
        final_edge_contours = []

        for c in remaining_edge_contours:
            final_edge_contours.append(cv2.convexHull(c))

        print("%d remaining edge candidates found" % len(final_edge_contours))

        all_contours.extend(final_edge_contours)

    print("%d total candidates found" % len(all_contours))

    return all_contours


def process_test_data(img_hsv, contours, saved_region_dir=None):
    print("Generating feature metrics...")

    test_data_processed = []

    for c_idx, c in enumerate(contours):
        if saved_region_dir is not None:
            region_base_name = '.'.join([str(c_idx), 'png'])
            region_file_path = os.path.join(saved_region_dir, region_base_name)

            if not os.path.exists(saved_region_dir):
                os.makedirs(saved_region_dir)
        else:
            region_file_path = None
        features = color_utils.generate_features(
            img_hsv,
            c,
            region_file_path=region_file_path
        )
        test_data_processed.append(features)

    test_data_processed = pd.DataFrame(test_data_processed)
    test_data_processed.drop('label', axis=1, inplace=True)

    return test_data_processed


def predict(test_data_processed, xgb_model, categories):
    print("Classifying regions...")
    pred_results = []

    d_test = xgb.DMatrix(test_data_processed)

    tmp_results = xgb_model.predict(d_test)

    for idx, probabilities in enumerate(tmp_results):

        # probabilities = xgb_model.predict(d_test)
        labelled_probs = {a: probabilities[i] for i, a in enumerate(categories)}
        pred_label = max(labelled_probs, key=lambda key: labelled_probs[key])
        pred_label_prob = labelled_probs[pred_label]

        pred_results.append(
            {
                'contour_index': idx,
                'prob': labelled_probs,
                'label': pred_label,
                'label_prob': pred_label_prob
            }
        )

    return pred_results


def plot_test_results(img_hsv, contours, pred_results, save_dir, annotate=True):
    print("Plotting results...")
    label_contours = {}

    for res in pred_results:
        label = res['label']

        if label not in label_contours:
            label_contours[label] = {
                'contour_indices': [],
                'contours': [],
                'prob': []
            }

        label_contours[label]['contour_indices'].append(res['contour_index'])
        label_contours[label]['prob'].append(res['label_prob'])
        label_contours[label]['contours'].append(contours[res['contour_index']])

    for label, c_dict in label_contours.items():
        # create image for class label
        new_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        cv2.drawContours(new_img, c_dict['contours'], -1, (0, 255, 0), 5)
        plt.figure(figsize=(16, 16))
        plt.imshow(new_img)
        plt.title("%s" % label)

        for i, c in enumerate(c_dict['contours']):
            x, y, w, h = cv2.boundingRect(c)
            idx = c_dict['contour_indices'][i]
            prob = c_dict['prob'][i]

            if annotate:
                plt.text(
                    x,
                    y,
                    "%d (%.2f)" % (idx, prob),
                    fontsize=12,
                    color='lime',
                    backgroundcolor='#00000088'
                )

        if save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, label + '.png'),
                bbox_inches='tight'
            )
        else:
            plt.show()
