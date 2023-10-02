import numpy as np
from datasets import Dataset
from typing import List, Tuple, Dict
import numpy as np
import cv2
from collections import namedtuple
import os
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


BoxRel = namedtuple('BoxRel', ['i', 'j', 'mag', 'projs'])


EDGE_COLOR_SET = ['red', 'green', 'blue', 'black', 'gray']


def dummy_calculate_relation_set(dataset: Dataset, k: int, clusters: int):
    """Default: return top, down, left, right"""
    return [(0, 1), (0, -1), (-1, 0), (1, 0)]


def angle_dist(a, b):
    # define a function to calculate the distance between angles
    return np.arccos(np.cos(a - b))

class AngleKMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers_ = None

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        
        # Randomly choose clusters
        i = rng.permutation(X.shape[0])[:self.n_clusters]
        self.centers_ = X[i]

        for _ in range(self.max_iter):
            # Assign labels based on closest center
            labels = pairwise_distances_argmin(X, self.centers_, metric=angle_dist)

            # Find new centers from means of points
            new_centers = np.array([X[labels == i].mean(0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.all(self.centers_ == new_centers):
                break

            self.centers_ = new_centers

        return self

    def predict(self, X):
        return pairwise_distances_argmin(X, self.centers_, metric=angle_dist)


def calculate_relation_set(dataset: Dataset, k: int, clusters: int) -> List[Tuple[float, float]]:
    """ Calculate the relation set using all samples in the dataset """
    # First, use knn to find the k nearest neighbors
    knn_rels = []
    for data in dataset:
        # First, calculate the center of each box
        centers = []
        for box in data['boxes']:
            centers.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        # Second, calculate the distance between each center
        dists = cdist(centers, centers)
        # Third, find the k nearest neighbors
        knn = NearestNeighbors(n_neighbors=k, metric='precomputed')
        knn.fit(dists)
        knn_rels.append(knn.kneighbors_graph().toarray())
    # translate all these relations to direction
    all_relation = []
    for i, data in enumerate(dataset):
        data_relations = []
        for j, box in enumerate(data['boxes']):
            for k, other_box in enumerate(data['boxes']):
                if j == k:
                    continue
                relation = np.array([other_box[0] - box[0], other_box[1] - box[1]])
                mag = np.linalg.norm(relation)
                dir_normed = relation / np.linalg.norm(mag)
                data_relations.append(dir_normed)
        all_relation.append(data_relations)
    # convert these relations to radians
    all_relation = np.array(all_relation)
    all_relation = np.arctan2(all_relation[:, :, 1], all_relation[:, :, 0])
    # use kmeans to cluster these relations
    # customized kmeans to use angle distance
    kmeans = AngleKMeans(n_clusters=clusters, random_state=0).fit(all_relation.reshape(-1, 1))
    # get the centers
    centers = kmeans.centers_.reshape(-1)
    # get the relations
    final_rels = []
    for i in range(len(centers)):
        # calculate vector from angle
        centers[i] = np.array([np.cos(centers[i]), np.sin(centers[i])])
        final_rels.append(centers[i])
    return final_rels


def filter_relation(data_relation: List[BoxRel], thres=0.9):
    ''' If two relations are close enough in terms of cosine simiarlity, then we keep only the one that has smaller distance
    Args:
        data_relation: a list of tuples [{(i, j, mag, dir_normed)}] (clusters is the number of kept relation in the above step)
    '''
    cnt = 0
    filtered_data_relation = []
    while cnt < len(data_relation):
        rels = [data_relation[cnt]]
        while data_relation[cnt][0] == rels[0][0]:
            cnt += 1
            rels.append(data_relation[cnt])
        # sort rels by dist
        rels = sorted(rels, key=lambda x: x.mag)
        # filter
        rm_idxs = []
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                if np.dot(rels[i].projs, rels[j].projs) > thres:
                    rm_idxs.append(j)
        rels = [rels[i] for i in range(len(rels)) if i not in rm_idxs]
        filtered_data_relation.extend(rels)
    return filtered_data_relation


def calculate_relation(dataset: Dataset, relation_set: List[Tuple[float, float]]) -> List[List[BoxRel]]:
    """ Calculate the relation between samples in the dataset """
    all_relation = []
    for i, data in enumerate(dataset):
        data_relations = []
        for j, box in enumerate(data['boxes']):
            for k, other_box in enumerate(data['boxes']):
                if j == k:
                    continue
                center1 = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                center2 = (other_box[0] + other_box[2]) / 2, (other_box[1] + other_box[3]) / 2
                relation = np.array([center2[0] - center1[0], center2[1] - center1[1]])
                mag = np.linalg.norm(relation)
                dir_normed = relation / np.linalg.norm(mag)
                relation_projection = np.dot(dir_normed, relation_set)
                data_relations.append(BoxRel(j, k, mag, relation_projection))
        filtered_data_relation = filter_relation(data_relations)
        all_relation.append(filtered_data_relation)
    return all_relation


def visualize_relation(dataset: Dataset, relation_set: List[Tuple[float, float]], output_path: str):
    """ Visualize the relation between samples in the dataset """
    all_relation = calculate_relation(dataset, relation_set)
    os.makedirs(output_path, exist_ok=True)
    for i, (data, data_relation) in enumerate(zip(dataset, all_relation)):
        # open the image path
        img_fp = data['image_path']
        img = cv2.imread(img_fp)
        for j, box in enumerate(data['boxes']):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
            # put label on top of the rectangle
            cv2.putText(img, str(data['label'][j]), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for j, k, mag, prj in data_relation:
            max_prj_idx = np.argmax(prj)
            color = EDGE_COLOR_SET[max_prj_idx]
            center_j = (data['boxes'][j][0] + data['boxes'][j][2]) / 2, (data['boxes'][j][1] + data['boxes'][j][3]) / 2
            center_k = (data['boxes'][k][0] + data['boxes'][k][2]) / 2, (data['boxes'][k][1] + data['boxes'][k][3]) / 2
            cv2.line(img, center_j, center_k, color, 2)
        cv2.imwrite(output_path + str(i) + '.jpg', img)


if __name__ == '__main__':
    import argparse
    import glob
    from training_layoutlmv3.utils import load_data
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='data', type=str, default="data/preprocessed",
                        help='folder of training data consisting of .json and .jpg files')
    args = parser.parse_args()
    train_data_dir = f"{args.data}/"
    dataset = [load_data(fp, f"{fp[:-5]}.jpg") for fp in glob.glob(f"{train_data_dir}/*.json")]
    relation_set = calculate_relation_set(dataset, k=10, clusters=8)
    visualize_relation(dataset, relation_set, 'relation_vis/')
