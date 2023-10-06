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
import tqdm
from collections import Counter


BoxRel = namedtuple('BoxRel', ['i', 'j', 'mag', 'projs'])


EDGE_COLOR_SET = ['red', 'green', 'blue', 'black', 'gray', 'yellow', 'pink', 'orange']
MAP_COLOR_TO_CV2_TUPLE = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'yellow': (0, 255, 255),
    'pink': (255, 0, 255),
    'orange': (0, 128, 255),
}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)

        if px == py:
            return False

        if self.size[px] > self.size[py]:
            px, py = py, px

        self.parent[px] = py
        self.size[py] += self.size[px]

        return True

    def groups(self):
        ans = [[] for _ in range(len(self.parent))]
        for i in range(len(self.parent)):
            ans[self.find(i)].append(i)
        return list(filter(lambda x: len(x) > 0, ans))




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
    bar = tqdm.tqdm(dataset, total=len(dataset))
    bar.set_description("Collecting all relations")
    all_relations = []
    for data in bar:
        # First, calculate the center of each box
        centers = np.array(list([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in data['boxes']))
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(centers)))
        nbrs.fit(centers)
        distances, indices = nbrs.kneighbors(centers)
        for i, idx_row in enumerate(indices):
            for idx in idx_row:
                if idx != i:  # Exclude self relations
                    relation = centers[idx] - centers[i]
                    dir_normed = relation / np.linalg.norm(relation)
                    all_relations.append(dir_normed)

    # Convert relations to angles
    all_relations = np.array(all_relations)
    angles = np.arctan2(all_relations[:, 1], all_relations[:, 0]).reshape(-1, 1)

    # 3. KMeans on Filtered Relations
    kmeans = AngleKMeans(n_clusters=clusters, random_state=0).fit(angles)
    centers = kmeans.centers_.squeeze()
    final_rels = np.column_stack([np.cos(centers), np.sin(centers)])
    
    return final_rels

def filter_relation(data_relation: List[BoxRel], relation_set, thres=0.5):
    ''' If two relations are close enough in terms of cosine simiarlity, then we keep only the one that has smaller distance
    Args:
        data_relation: a list of tuples [{(i, j, mag, dir_normed)}] (clusters is the number of kept relation in the above step)
    '''
    cnt = 0
    filtered_data_relation = []
    while cnt < len(data_relation):
        rels = [data_relation[cnt]]
        while cnt < len(data_relation) and data_relation[cnt][0] == rels[0][0]:
            rels.append(data_relation[cnt])
            cnt += 1
        # sort rels by dist
        rels = sorted(rels, key=lambda x: x.mag)
        # filter
        rm_idxs = []
        # reconstruct the original vector based on matmul proj and relation set
        proj = np.array([rel.projs for rel in rels]) # (n, clusters)
        # relation_set is (clusters, 2)
        # proj is (n, clusters)
        # proj @ relation_set is (n, 2)
        vecs = proj @ relation_set
        # convert to radians
        angls = np.arctan2(vecs[:, 1], vecs[:, 0])
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                if angle_dist(angls[i], angls[j]) < thres:
                    rm_idxs.append(j)
        rels = [rels[i] for i in range(len(rels)) if i not in rm_idxs]
        filtered_data_relation.extend(rels)
    return filtered_data_relation


def filter_relation_med_dist(data_relation: List[BoxRel], coeff=1.2):
    # calculate the median of the distance
    dists = [rel.mag for rel in data_relation]
    dists = np.array(dists)
    med_dist = np.median(dists)
    # filter
    filtered_data_relation = [rel for rel in data_relation if rel.mag < coeff * med_dist]
    return filtered_data_relation


def group_close_horizontal_words(data_relation: List[BoxRel], data, relation_set, thres=1.2):
    ''' If two relations are close enough in terms of cosine simiarlity, then we keep only the one that has smaller distance
    Args:
        data_relation: a list of tuples [{(i, j, mag, dir_normed)}] (clusters is the number of kept relation in the above step)
    '''
    # 1. Find all horizontal relations and mags
    horizontal_rels = []
    mags = []
    # reconstruct the original vector based on matmul proj and relation set
    proj = np.array([rel.projs for rel in data_relation]) # (n, clusters)
    # relation_set is (clusters, 2)
    # proj is (n, clusters)
    # proj @ relation_set is (n, 2)
    vecs = proj @ relation_set
    # convert to radians
    angls = np.arctan2(vecs[:, 1], vecs[:, 0])
    for i, rel in enumerate(data_relation):
        if angle_dist(angls[i], 0) < 0.1:
            horizontal_rels.append(rel)
            if data['boxes'][rel.i][0] < data['boxes'][rel.j][0]:
                mags.append(max(data['boxes'][rel.j][0] - data['boxes'][rel.i][2], 0))
            else:
                mags.append(max(data['boxes'][rel.i][0] - data['boxes'][rel.j][2], 0))
    med_mag = np.median(mags)
    # union find
    # for all horizontal relations, if the mag is smaller than the median * thres, we add them to UF.
    uf = UnionFind(len(data['boxes']))
    for i, rel in enumerate(horizontal_rels):
        if mags[i] < med_mag * thres:
            uf.union(rel.i, rel.j)
    # For each group, merge the boxes and text
    boxes = []
    words = []
    label = []
    for group in uf.groups():
        # merge boxes
        group_box = np.array(list([data['boxes'][i] for i in group]))
        boxes.append(np.array([np.min(group_box[:, 0]), np.min(group_box[:, 1]), np.max(group_box[:, 2]), np.max(group_box[:, 3])]))
        # merge words
        group_word = sorted(list([(i, data['words'][i]) for i in group]),key=lambda x: data['boxes'][int(x[0])][0])
        words.append(' '.join([word for _, word in group_word]))
        # merge label
        group_label = Counter([data['label'][i] for i in group])
        label.append(group_label.most_common(1)[0][0])
    data['boxes'] = boxes
    data['words'] = words
    data['label'] = label
    data_relation = calculate_relation([data], relation_set)[0]
    return data, data_relation



def calculate_relation(dataset: Dataset, relation_set: List[Tuple[float, float]]) -> List[List[BoxRel]]:
    """ Calculate the relation between samples in the dataset """
    all_relation = []

    relation_set = np.array(relation_set)

    for i, data in enumerate(dataset):
        boxes = np.array(list(data['boxes']))
        centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])

        # Using broadcasting to compute pairwise differences
        differences = centers[None, :, :] - centers[:, None, :]
        magnitudes = np.linalg.norm(differences, axis=2)

        # Mask to remove relations of a box with itself
        mask = np.eye(len(centers), dtype=bool)
        
        # Normalizing the differences
        with np.errstate(divide='ignore', invalid='ignore'):
            dir_normed = differences / magnitudes[..., None]
            dir_normed[mask] = 0

        # Projecting the relations onto the relation_set
        relation_projections = np.einsum('ijk,kl->ijl', dir_normed, relation_set.T)
        
        data_relations = []
        for j in range(len(centers)):
            for k in range(len(centers)):
                if j != k:
                    data_relations.append(BoxRel(j, k, magnitudes[j, k], relation_projections[j, k]))
        
        filtered_data_relation = data_relations
        filtered_data_relation = filter_relation(data_relations, relation_set)
        all_relation.append(filtered_data_relation)

    return all_relation




def visualize_relation(dataset: Dataset, relation_set: List[Tuple[float, float]], output_path: str):
    """ Visualize the relation between samples in the dataset """
    all_relation = calculate_relation(dataset, relation_set)
    os.makedirs(output_path, exist_ok=True)
    for i, (data, data_relation) in enumerate(zip(dataset, all_relation)):
        data, data_relation = group_close_horizontal_words(data_relation, data, relation_set)
        data_relation = filter_relation_med_dist(data_relation)
        # open the image path
        img_fp = data['image_path']
        img = cv2.imread(img_fp)
        for j, box in enumerate(data['boxes']):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
            # put label on top of the rectangle
            # cv2.putText(img, str(data['label'][j]), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # for j, k, mag, prj in data_relation:
        #     max_prj_idx = np.argmax(prj)
        #     color = MAP_COLOR_TO_CV2_TUPLE[EDGE_COLOR_SET[max_prj_idx]]
        #     center_j = (data['boxes'][j][0] + data['boxes'][j][2]) / 2, (data['boxes'][j][1] + data['boxes'][j][3]) / 2
        #     center_k = (data['boxes'][k][0] + data['boxes'][k][2]) / 2, (data['boxes'][k][1] + data['boxes'][k][3]) / 2
        #     cv2.line(img, (int(center_j[0]), int(center_j[1])), (int(center_k[0]), int(center_k[1])), color, 2)
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
    relation_set = dummy_calculate_relation_set(dataset, 10, 4)
    # if os.path.exists('relation_set.npy'):
    #     relation_set = np.load('relation_set.npy')
    # else:
    #     relation_set = calculate_relation_set(dataset, k=10, clusters=8)
    #     np.save('relation_set.npy', relation_set)
    visualize_relation(dataset, relation_set, 'relation_vis/')
