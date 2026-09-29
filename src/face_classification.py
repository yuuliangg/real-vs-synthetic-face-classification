"""Real vs Synthetic Face Classification.

Portfolio-friendly version of the original coursework project.
Compares HOG and raw-colour features with KNN and linear SVM.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


IMG_SIZE = (128, 128)
CLASS_NAMES = ["Real", "Synthetic"]
LABEL_MAP = {0: "Real", 1: "Synthetic"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REAL_DIR = DATA_DIR / "RealFace"
SYN_DIR = DATA_DIR / "SynFace"
TEST_REAL_DIR = DATA_DIR / "test_new" / "RealFace"
TEST_SYN_DIR = DATA_DIR / "test_new" / "SynFace"


def image_files(folder: Path):
    """Return supported image files in a folder in stable order."""
    if not folder.exists():
        raise FileNotFoundError(
            f"Missing dataset folder: {folder}\n"
            "See data/README.md for the expected folder structure."
        )
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def process_image_hog(image_path: Path, size=IMG_SIZE):
    """Resize an image, convert it to grayscale and extract HOG features."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, size) / 255.0
    return hog(
        image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )


def process_image_raw(image_path: Path, size=IMG_SIZE):
    """Resize an image, normalise colour values and flatten the pixels."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.resize(image, size) / 255.0
    return image.flatten()


def load_training_data():
    """Load real and synthetic training images into HOG/raw feature arrays."""
    x_hog, x_raw, y = [], [], []
    folders = {0: REAL_DIR, 1: SYN_DIR}

    for label, folder in folders.items():
        for path in image_files(folder):
            hog_features = process_image_hog(path)
            raw_features = process_image_raw(path)

            if hog_features is not None and raw_features is not None:
                x_hog.append(hog_features)
                x_raw.append(raw_features)
                y.append(label)

    return np.asarray(x_hog), np.asarray(x_raw), np.asarray(y)


def show_feature_examples():
    """Display original, grayscale and HOG views for one image per class."""
    samples = {
        "Real": image_files(REAL_DIR)[0],
        "Synthetic": image_files(SYN_DIR)[0],
    }

    plt.figure(figsize=(10, 7))
    plot_position = 1

    for name, path in samples.items():
        colour = cv2.imread(str(path))
        colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        gray = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), IMG_SIZE)

        _, hog_image = hog(
            gray / 255.0,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
        )

        plt.subplot(2, 3, plot_position)
        plt.imshow(colour)
        plt.title(f"{name} - Original")
        plt.axis("off")

        plt.subplot(2, 3, plot_position + 1)
        plt.imshow(gray, cmap="gray")
        plt.title(f"{name} - Grayscale")
        plt.axis("off")

        plt.subplot(2, 3, plot_position + 2)
        plt.imshow(hog_image, cmap="gray")
        plt.title(f"{name} - HOG")
        plt.axis("off")

        plot_position += 3

    plt.tight_layout()
    plt.show()


def run_pipeline(x, y, feature_name):
    """Tune KNN, train KNN/SVM and show held-out test results."""
    print(f"\n========== {feature_name} ==========")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print("Train:", x_train.shape, "Test:", x_test.shape)

    k_range = list(range(1, 50, 2))
    mean_scores = []

    for k in k_range:
        knn_cv = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(
            knn_cv,
            x_train,
            y_train,
            cv=10,
            scoring="accuracy",
        )
        mean_scores.append(scores.mean())

    best_k = k_range[int(np.argmax(mean_scores))]
    print(f"Best K: {best_k}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, mean_scores, marker="o")
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.title(f"{feature_name} - Cross-Validation Accuracy vs K")
    plt.xticks(k_range)
    plt.grid()
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)
    knn_predictions = knn.predict(x_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)

    svm = SVC(kernel="linear", class_weight="balanced")
    svm.fit(x_train, y_train)
    svm_predictions = svm.predict(x_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    print(f"KNN test accuracy: {knn_accuracy * 100:.2f}%")
    print(f"SVM test accuracy: {svm_accuracy * 100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, model_name, predictions in zip(
        axes,
        ["KNN", "SVM"],
        [knn_predictions, svm_predictions],
    ):
        matrix = confusion_matrix(y_test, predictions)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )
        ax.set_title(f"{feature_name} - {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

    return knn, svm, knn_accuracy, svm_accuracy


def compare_cross_validation(models, y):
    """Compare the four final model/feature combinations with five-fold CV."""
    results = {}

    for name, (model, x) in models.items():
        scores = cross_val_score(model, x, y, cv=5, scoring="accuracy")
        results[name] = scores
        print(f"{name}: {scores}")
        print(
            f"Mean = {scores.mean() * 100:.2f}% "
            f"Std = {scores.std() * 100:.2f}%\n"
        )

    best_mean = max(results, key=lambda name: results[name].mean())
    best_std = min(results, key=lambda name: results[name].std())

    print(
        f"Highest mean accuracy: {best_mean} "
        f"({results[best_mean].mean() * 100:.2f}%)"
    )
    print(
        f"Most consistent: {best_std} "
        f"(std {results[best_std].std() * 100:.2f}%)"
    )

    return results


def evaluate_external_images(hog_knn, hog_svm, raw_knn, raw_svm):
    """Evaluate the external set and return aggregate accuracy only.

    Individual filenames, predictions and images are intentionally not displayed
    in the public portfolio version.
    """
    test_folders = {0: TEST_REAL_DIR, 1: TEST_SYN_DIR}
    correct = {"HOG-KNN": 0, "HOG-SVM": 0, "RAW-KNN": 0, "RAW-SVM": 0}
    total = 0

    for true_label, folder in test_folders.items():
        for path in image_files(folder):
            hog_features = process_image_hog(path)
            raw_features = process_image_raw(path)
            if hog_features is None or raw_features is None:
                continue

            hog_features = hog_features.reshape(1, -1)
            raw_features = raw_features.reshape(1, -1)

            predictions = {
                "HOG-KNN": hog_knn.predict(hog_features)[0],
                "HOG-SVM": hog_svm.predict(hog_features)[0],
                "RAW-KNN": raw_knn.predict(raw_features)[0],
                "RAW-SVM": raw_svm.predict(raw_features)[0],
            }

            total += 1
            for name, prediction in predictions.items():
                correct[name] += int(prediction == true_label)

    accuracies = {}
    print(f"\nExternal-image accuracy on {total} images:")
    for name, count in correct.items():
        accuracy = count / total if total else 0.0
        accuracies[name] = accuracy
        print(f"{name}: {count}/{total} = {accuracy * 100:.2f}%")

    return accuracies


def main():
    show_feature_examples()
    x_hog, x_raw, y = load_training_data()

    print("HOG feature shape:", x_hog.shape)
    print("Raw feature shape:", x_raw.shape)
    print("Labels shape:", y.shape)
    print("Class balance -> Real:", np.sum(y == 0), "Synthetic:", np.sum(y == 1))

    hog_knn, hog_svm, hog_knn_acc, hog_svm_acc = run_pipeline(
        x_hog, y, "HOG features"
    )
    raw_knn, raw_svm, raw_knn_acc, raw_svm_acc = run_pipeline(
        x_raw, y, "Raw colour pixels"
    )

    print("\n================ SUMMARY ================")
    print(f"HOG -> KNN: {hog_knn_acc * 100:.2f}% SVM: {hog_svm_acc * 100:.2f}%")
    print(f"Raw -> KNN: {raw_knn_acc * 100:.2f}% SVM: {raw_svm_acc * 100:.2f}%")

    models = {
        "HOG-KNN": (hog_knn, x_hog),
        "HOG-SVM": (hog_svm, x_hog),
        "RAW-KNN": (raw_knn, x_raw),
        "RAW-SVM": (raw_svm, x_raw),
    }
    compare_cross_validation(models, y)
    evaluate_external_images(hog_knn, hog_svm, raw_knn, raw_svm)


if __name__ == "__main__":
    main()
