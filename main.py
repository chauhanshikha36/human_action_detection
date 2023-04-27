import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip

#------------------------------------------------------------------------------
#   Keypoints using MP Holistic
#------------------------------------------------------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

pose=[]
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])
#------------------------------------------------------------------------------
DATA_PATH = os.path.join('Activity_Data')
actions = np.array(['clapping', 'hands_up', 'punching', 'standing', 'waving', 'walking', 'sitting'])
no_sequences = 30
sequence_length = 30
label_map = {label:num for num, label in enumerate(actions)}
print("\nActivity:- ",label_map)
print('-------------------------------------')

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#------------------------------------------------------------------------------
#   LSTM Neural Network
#------------------------------------------------------------------------------
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('7model.h5')
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
colors = [(245,117,16), (117,245,16), (16,117,245),(245,117,16),(117,245,16),(16,117,245),(245,117,16)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame
#------------------------------------------------------------------------------
sequence = []
sentence = []
threshold = 0.98
temp = []
i=0
cap = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'XVID')
#recording_flag = False
act_flag = 1
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame,(1020,720))
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        cv2.rectangle(image, (0, 0), (1020, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'PLEASE let the camera detect whole human body!!', (10, 710), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            # Viz logic
            if res[np.argmax(res)] >= threshold:
                if len(sentence) > 0:
                    temp.append(actions[np.argmax(res)])
                    # cv2.imwrite(filename='images/' + str(i) + ' '.join(sentence) + '.jpg', img=image)

                    if actions[np.argmax(res)] != sentence[-1]:
                        act_flag =1;
                        #sentence.clear()
                        #sentence.append(actions[np.argmax(res)])


                    if (act_flag == 1):
                        if (len(temp) > 5 and actions[np.argmax(res)] == temp[-1] and actions[np.argmax(res)] == temp[
                            -2] and actions[np.argmax(res)] == temp[-3] and actions[np.argmax(res)] == temp[-4] and
                                actions[np.argmax(res)] == temp[-5]):
                            sentence.clear()
                            sentence.append(actions[np.argmax(res)])
                            act_flag = 0;
                        # if recording_flag == False:
                        #     output = cv2.VideoWriter('images/' + str(i) + ' '.join(sentence) + '.avi', codec, 7, (640, 480))
                        #     i = i + 1
                        #     recording_flag = True
                        # else:
                        #     recording_flag = False
                else:
                    sentence.clear()
                    sentence.append(actions[np.argmax(res)])

            #if len(sentence) > 3:
            #    sentence = sentence[-3:]

            # Viz probabilities
            #image = prob_viz(res, actions, image, colors)

        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(temp)
            print(sentence)
            break
        # if recording_flag:
        #     output.write(image)
            #path_to_video = 'images/' + str(i) + ' '.join(sentence) + '.avi'
            #clip = VideoFileClip(path_to_video)
            #duration = clip.duration
            #clip.close()
            #if duration < 3:
            #   os.remove(path_to_video)

    cap.release()
    cv2.destroyAllWindows()

