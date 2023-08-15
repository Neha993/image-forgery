from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import numpy as np
from progressbar import progressbar
import matplotlib.pyplot as plt
import glob
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image


def extract_chromatic_channel(bgr_img):
    # Extract 2 chromatic channes from BGR image
    # Input: BGR Image
    # Output: CrCb channels
    ycrcb_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCR_CB)

    return ycrcb_image[:, :, 1:]


def block_processing(cb_image, block_size, stride):
    # Divide image into multiple overlap blocks
    # Input: Cr or Cb channel
    # Output: List of blocks
    height, width, _ = cb_image.shape
    img_blocks = []
    for i in range(0, height - block_size, stride):
        for j in range(0, width - block_size, stride):
            img_blocks.append(cb_image[i: i + block_size, \
                              j: j + block_size])
    return np.array(img_blocks)


def extract_lbp_dct(blocks, n_points=8, radius=1):
    # Extract feature vector from given blocks
    # Input: List of blocks response with given image
    # Output: Feature vector of given image
    n_blocks, block_size, _, _ = blocks.shape
    CR_feature = np.zeros((n_blocks, block_size, block_size))
    CB_feature = np.zeros((n_blocks, block_size, block_size))
    for idx, block in enumerate(blocks):
        CR_lbp = local_binary_pattern(block[:, :, 0], n_points, radius)
        CR_lbp = np.float32(CR_lbp)
        # CR_lbp is a 2D vector
        CR_feature[idx] = cv2.dct(CR_lbp)
        # sv2.dct will return a 2d vector
        CB_lbp = local_binary_pattern(block[:, :, 1], n_points, radius)
        CB_lbp = np.float32(CB_lbp)
        CB_feature[idx] = cv2.dct(CB_lbp)
    CR_feature = np.std(CR_feature, axis=0).flatten()
    CB_feature = np.std(CB_feature, axis=0).flatten()

    return np.concatenate([CR_feature, CB_feature], axis=0)


def extract_feature(cb_image, block_size, stride):
    # Extract feature from given CrCb channels
    # Input: CrCb channels
    # Output: Feature vector or given original image
    img_blocks = block_processing(cb_image, block_size, stride)

    feature = extract_lbp_dct(img_blocks)
    return feature


def read_and_extract_feature(list_img, block_sizes, strides):
    # Read and extract feature vector from given list images
    total_img = len(list_img)
    dim = 0
    print(block_sizes[0])
    for i in range(len(block_sizes)):
        dim += block_sizes[i] ** 2
    features = np.zeros((total_img, 2 * dim))
    # 2d vector each index will contain feature of a image
    for idx in progressbar(range(len(list_img))):
        im = list_img[idx]
        bgr_img = cv2.imread(im)
        # bgr_img contain rgb value of image im
        cb_image = extract_chromatic_channel(bgr_img)
        tmp = 0
        for i, bz in enumerate(block_sizes):
            features[idx, tmp: tmp + 2 * bz ** 2] = extract_feature(cb_image, bz, strides[i])
            tmp += 2 * bz ** 2
    return features


def process_dataset(folders_real, folders_fake, block_sizes=[32], strides=[16]):
    # Process CASIA dataset
    # Label: 0 - fake image
    #        1 - real image
    list_real = []
    list_fake = []
    for fdr in folders_real:
        list_real += glob.glob(fdr)
    for fdf in folders_fake:
        list_fake += glob.glob(fdf)
    Y_train = np.zeros((len(list_real) + len(list_fake),), dtype=np.float32)
    Y_train[: len(list_real)] = 1.0
    X_train = read_and_extract_feature(list_real + list_fake, block_sizes=block_sizes, strides=strides)

    return X_train, Y_train


if __name__ == '__main__':
    folder_real = ['CASIA2/Au/*.jpg']
    folder_fake = ['CASIA2/Tp/*.jpg', 'CASIA2/Tp/*.tif']
    print('Build SVM model ...')
    X, Y = process_dataset(folder_real, folder_fake)
    print('Build SVM model ...')
    X, Y = shuffle(X, Y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = LinearSVC()
    clf.fit(X_train, Y_train)


    test_path = 'image2.png'
    X = read_and_extract_feature(glob.glob(test_path), block_sizes=[32], strides=[16])
    X = scaler.transform(X)
    predictSingleImage = clf.predict(X)
    print(predictSingleImage)

    Y = clf.predict(X_test)  #this is list of 0 and 1
    # print(Y)


    def calculate_metrics(confusion_matrix):
        # Calculate true positives, false positives, false negatives, and true negatives
        tp = confusion_matrix[1][1]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        tn = confusion_matrix[0][0]

        # Calculate precision
        precision = tp / (tp + fp)

        # Calculate accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        # Calculate recall
        recall = tp / (tp + fn)

        # Calculate F1 score
        f1_score = (2 * precision * recall) / (precision + recall)

        return precision, accuracy, recall, f1_score




    cm = confusion_matrix(Y_test, Y, labels=clf.classes_)
    print("Confusion Matrix: ")
    print(cm)
    precision, accuracy, recall, f1_score = calculate_metrics(cm)

    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    # this will plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
 # Custum input GUI




def browse_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])

    # Check if a file was selected
    if file_path:
        # Open and display the image using PIL
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.LANCZOS)

        # Convert the image to Tkinter-compatible format
        tk_image = ImageTk.PhotoImage(image)

        # Update the image label with the selected image
        image_label.configure(image=tk_image)
        image_label.image = tk_image
        Xs = read_and_extract_feature(glob.glob(file_path), block_sizes=[32], strides=[16])
        Xs = scaler.transform(Xs)
        predictSingle = clf.predict(Xs)
        print(predictSingle)
        global my_variable
        my_variable = predictSingle
        # update_variable(predictSingle)

def update_variable():
    # Update the variable's value



    # Update the label with the new value
    if my_variable :
        variable_label.set(str("Real Image!"))
    else:
        variable_label.set(str("Forged Image!"))



# Create the main window
window = tk.Tk()
window.title("Image Browser")
window.geometry("420x420")


# Create a button for browsing an image
browse_button = tk.Button(window, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)


# Create a label to display the image
image_label = tk.Label(window)
image_label.pack()

# create a button to show result
result_button = tk.Button(window, text="Show Result", command=update_variable)
result_button.pack(pady=10)

# --------------------------------
my_variable = ""

# Create a StringVar and set its initial value
variable_label = tk.StringVar()
variable_label.set(str(my_variable))


label = tk.Label(window, textvariable=variable_label, font=("Arial", 18))
label.pack(pady=10)
# ----------------------------

# Start the GUI event loop
window.mainloop()
