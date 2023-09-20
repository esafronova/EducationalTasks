import os
from os import environ
from os.path import join
from sys import argv


def save_csv(img_classes, filename):
    with open(filename, 'w') as fhandle:
        print('filename,class_id', file=fhandle)
        for filename in sorted(img_classes.keys()):
            print('%s,%s' % (filename, img_classes[filename]), file=fhandle)


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = (class_id)
    return res


def check_test(output_dir, gt_dir):
    img_output = read_csv(join(output_dir, 'image_output.csv'))
    img_gt = read_csv(join(gt_dir, 'image_gt.csv'))

    vid_output = read_csv(join(output_dir, 'video_output.csv'))
    vid_gt = read_csv(join(gt_dir, 'video_gt.csv'))

    correct = 0.
    total = len(img_gt)
    for k, v in img_gt.items():
        if img_output[k] == v:
            correct += 1
    accuracy_1 = correct / total

    correct = 0.
    total = len(vid_gt)
    for k, v in vid_gt.items():
        if vid_output[k] == v:
            correct += 1
        else:
            print('classified: ', vid_output[k], ' should be: ',v)
    accuracy_2 = correct / total

    return 'Ok, accuracies %.4f, %.4f' % (accuracy_1, accuracy_2)


def grade(results_list):
    test_data_result = results_list[-1]

    result = test_data_result['result']
    if not result.startswith('Ok'):
        return '', 0

    accuracy_str = result[15:].split(', ')
    accuracy_img = float(accuracy_str[0])

    if accuracy_img >= 0.977:
        mark_img = 10
    elif accuracy_img >= 0.971:
        mark_img = 9
    elif accuracy_img >= 0.965:
        mark_img = 8
    elif accuracy_img >= 0.955:
        mark_img = 6
    elif accuracy_img >= 0.946:
        mark_img = 5
    elif accuracy_img >= 0.93:
        mark_img = 4
    elif accuracy_img >= 0.865:
        mark_img = 2
    else:
        mark_img = 0

    accuracy_video = float(accuracy_str[1])

    if accuracy_video >= 0.90:
        mark_video = 10
    elif accuracy_video >= 0.86:
        mark_video = 8
    elif accuracy_video >= 0.82:
        mark_video = 6
    elif accuracy_video >= 0.76:
        mark_video = 4
    elif accuracy_video >= 0.70:
        mark_video = 2
    else:
        mark_video = 0

    mark = round(2 * mark_img / 3. + mark_video / 3)
    return (accuracy_img, accuracy_video), mark


def run_single_test(data_dir, output_dir):
    from classification_face import Classifier, preprocess_imgs
    from keras import backend as K
    from keras.models import load_model
    from os import environ
    from os.path import abspath, dirname, join

    img_train_dir = join(data_dir, 'img_train')
    img_test_dir = join(data_dir, 'img_test')

    img_train_gt = read_csv(join(img_train_dir, 'gt.csv'))
    img_train_images_dir = join(img_train_dir, 'images')

    nn_path = join(data_dir, 'face_recognition_model.h5')
    cascade_path = join(data_dir, 'haarcascade_frontalface_default.xml')
    clf = Classifier(load_model(nn_path), cascade_path)
    clf.fit(img_train_images_dir, img_train_gt, 'images_train_fit.npy')
    img_test_images_dir = join(img_test_dir, 'images')
    img_classes = clf.classify_images(img_test_images_dir)
    save_csv(img_classes, join(output_dir, 'image_output.csv'))
    if environ.get('KERAS_BACKEND') == 'tensorflow':
        K.clear_session()

    vid_train_dir = join(data_dir, 'vid_train')
    vid_test_dir = join(data_dir, 'vid_test')

    vid_train_gt = read_csv(join(vid_train_dir, 'gt.csv'))
    vid_train_images_dir = join(vid_train_dir, 'images')

    clf = Classifier(load_model(nn_path), cascade_path)
    clf.fit(vid_train_images_dir, vid_train_gt, 'video_train_fit.npy')
    vid_test_videos_dir = join(vid_test_dir, 'videos')
    video_classes = clf.classify_videos(vid_test_videos_dir)
    save_csv(video_classes, join(output_dir, 'video_output.csv'))
    if environ.get('KERAS_BACKEND') == 'tensorflow':
        K.clear_session()


if __name__ == '__main__':
    from glob import glob
    from re import sub
    from time import time
    from traceback import format_exc
    from os import makedirs

    tests_dir = r'C:\Users\Ekaterina\Documents\ImageProcess\public_data\public_data_classification'

    results = []
    for input_dir in sorted(glob(join(tests_dir, '[0-9][0-9]_input'))):
        output_dir = sub('input$', 'output', input_dir)
        # if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)
        makedirs(output_dir, exist_ok=True)
        gt_dir = sub('input$', 'gt', input_dir)

        try:
            start = time()
            run_single_test(input_dir, output_dir)
            end = time()
            running_time = end - start
        except:
            result = 'Runtime error'
            traceback = format_exc()
        else:
            try:
                result = check_test(output_dir, gt_dir)
            except:
                result = 'Checker error'
                traceback = format_exc()

        test_num = input_dir[-8:-6]
        if result == 'Runtime error' or result == 'Checker error':
            print(test_num, result, '\n', traceback)
            results.append({'result': result})
        else:
            print(test_num, '%.2fs' % running_time, result)
            results.append({
                'time': running_time,
                'result': result})

    description, mark = grade(results)
    print('Mark:', mark, 'image accuracy:', description[0], 'video accuracy:', description[1])