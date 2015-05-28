__author__ = 'KranthiDhanala'
import cv2 as cv
import numpy as np
from sklearn.multiclass import OneVsRestClassifier,OutputCodeClassifier
from sklearn.svm import SVC
from os.path import isfile
import random as rand


MHI_DURATION = 1
DEFAULT_THRESHOLD = 32

def main():
    """
    the folder path which contains all the video files
    """
    def clock():
        return cv.getTickCount() / cv.getTickFrequency()
    #containing paths to test and train folders
    train_folder_path = "/train.txt"
    test_folder_path = "/test.txt"
    if (not isfile(train_folder_path)) or (not isfile(test_folder_path)):
        print "please enter correct folder path!!"
        exit(1)

    """
    get all the file folders names in folder which specify the video category and video path
    """
    label_train = []
    label_test = []
    train_videos = []
    test_videos =[]
    print "Reading Text Files!!"
    with open(train_folder_path) as train_file:
        for line in train_file:
            if len(line):
                lbl,video_path = line.split()
                train_videos.append(video_path)
                label_train.append(int(lbl))
    with open(test_folder_path) as test_file:
        for line in test_file:
            if len(line) > 1:
                lbl,video_path = line.split()
                test_videos.append("D:/videoanalytics/"+video_path)
                label_test.append(int(lbl))


    training_data=[]
    testing_data = []
    train_count = len(label_train)
    count_nbr = 0
    """
    in each category retrieve all the video files and extract features from each video
    """

    print "CONSTRUCTING FEATURES FOR TRAINING AND TESTING DATA!!"
    for each_video_path in train_videos+test_videos:
        if not isfile(each_video_path):
            print "Video path doesn't exist!!",each_video_path
            exit(1)
        count_nbr = count_nbr + 1
        video = cv.VideoCapture(each_video_path)
        if not video.isOpened():
            print "video cannot be opened"
            exit(1)

            #read the first frame (t =1)
        ret, frame = video.read()
        h, w = frame.shape[:2]
        prev_frame = frame.copy()
        motion_history = np.zeros((h, w), np.float32)
        timestamp = clock()
        vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
        fram_count = 1
        while(video.isOpened()):
            ret, frame = video.read()
            if ret == False:
                #to see the mhi image uncomment following line
                #cv.imshow('Motion History Image',vis)
                vis = cv.cvtColor(vis,cv.COLOR_BGR2GRAY)

                features = cv.HuMoments(cv.moments(vis)).flatten()
                features = features.reshape((1,7))
                features = features.reshape(-1)
                features = features.tolist()
                features= -np.sign(features) * np.log10(np.abs(features))
                if count_nbr <= train_count:
                    training_data.append(features)
                else:
                    testing_data.append(features)

                #to see the mhi image uncomment following lines
                #if cv.waitKey(30) & 0xFF == ord('q'):
                #    break
                break

            if fram_count%rand.randint(1,7) ==0 :
                frame_diff = cv.absdiff(frame, prev_frame)
                gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
                ret, motion_mask = cv.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv.THRESH_BINARY)

                cv.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
                vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
                vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
            fram_count = fram_count+1
            prev_frame = frame.copy()

        video.release()
        cv.destroyAllWindows()

    """
    using rbf kernel with degree 3 will
    """

    print "TRAINING CLASSIFER"
    clf = OneVsRestClassifier(SVC(random_state=0,kernel="sigmoid"))
    clf.fit(np.array(training_data), np.array(label_train))
    print "TESTING ON TRAINING DATA"
    predict_train = clf.predict(training_data)

    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    fram_count = 0
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = clock()
    vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        if fram_count%rand.randint(1,7) ==0 :
                frame_diff = cv.absdiff(frame, prev_frame)
                gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
                ret, motion_mask = cv.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv.THRESH_BINARY)

                cv.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
                vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
        fram_count = fram_count+1
        if fram_count%300 == 0:
            # classify the motion history image and classify the motion history image
            features = cv.HuMoments(cv.moments(vis)).flatten()
            features = features.reshape((1,7))
            features = features.reshape(-1)
            features = features.tolist()
            features = -np.sign(features) * np.log10(np.abs(features))
            predict_test = clf.predict(features)
            print "motion is :", predict_test
            #reset all back
            h, w = frame.shape[:2]
            prev_frame = frame.copy()
            motion_history = np.zeros((h, w), np.float32)
            timestamp = clock()
            vis = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)


        prev_frame = frame.copy()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print "TESTING ON TESTING DATA"
    predict_test = clf.predict(testing_data)

    correct_train = 0.0
    false_train = 0.0
    for i in range(len(label_train)):
        print label_train[i],predict_train[i]
        if label_train[i] == predict_train[i]:
            correct_train = correct_train+1
        else:
            false_train = false_train+1
    print "Totally Classified training data"
    print correct_train,false_train,correct_train/(float)(correct_train+false_train) * 100
    correct_cat = {}
    false_cat = {}
    for i in range(len(label_test)):
        correct_cat[i] = 0.0
        false_cat[i] = 0.0

    correct_test = 0.0
    false_test = 0.0

    for i in range(len(label_test)):
        print label_test[i],predict_test[i]
        if label_test[i] == predict_test[i]:
            correct_test = correct_test+1
            if correct_cat.has_key(label_test[i]):
                correct_cat[label_test[i]] = correct_cat[label_test[i]] + 1

        else:
            false_test = false_test+1
            if false_cat.has_key(label_test[i]):
                false_cat[label_test[i]] = false_cat[label_test[i]] + 1


    print "label    correct   wrong     percentage"
    for i in range(1,11):
        if correct_cat[i] != 0 and false_cat[i] !=0:
            print i,"       ",correct_cat[i],"       ",false_cat[i],"           ",correct_cat[i]/(correct_cat[i]+false_cat[i])*100


    print "Totally Classified testing data"
    print "Correct ",correct_test,"False ",false_test,"Percentage ",correct_test/(correct_test+false_test) * 100


    return


if __name__ == '__main__':
    main()