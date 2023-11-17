import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import cv2
import os

def extract_nodes_relations_and_constraints(program):
    w_nodes = re.findall(r'(w\d+)', program)
    w_labels = re.findall(r'(w\d+).label == \'(L_\w+)\'', program)
    relations = re.findall(r'rel\((w\d+), (r\d+), (w\d+)\)', program)
    r_labels = re.findall(r'(r\d+).lbl == \'(L_\d+)\'', program)
    return w_nodes, w_labels, relations, r_labels

def adjust_pos(pos, direction):
    x, y = pos
    if direction == 'L_0':  # top
        return (x, y - 1)
    if direction == 'L_1':  # down
        return (x, y + 1)
    if direction == 'L_2':  # left
        return (x - 1, y)
    if direction == 'L_3':  # right
        return (x + 1, y)
    return pos

def visualize_program(program, out_path):
    G = nx.DiGraph()

    w_nodes, w_labels, relations, r_labels = extract_nodes_relations_and_constraints(program)

    for node in w_nodes:
        label = dict(w_labels).get(node, node)  # Use the label if available, otherwise use the node name
        G.add_node(node, label=label)

    mapping = {
        'L_0': 'top',
        'L_1': 'down',
        'L_2': 'left',
        'L_3': 'right'
    }

    for source, edge_label, target in relations:
        edge_constraint = dict(r_labels).get(edge_label, '')
        G.add_edge(source, target, label=edge_label + '-' + edge_constraint)

    # Initial positioning of nodes
    pos = nx.spring_layout(G)

    # Adjusting position based on edge labels
    for source, target in G.edges():
        edge_label = G[source][target]['label'].split('-')[1]
        pos[target] = adjust_pos(pos[source], edge_label)

    nx.draw(G, pos, with_labels=False, node_size=5000, node_color='skyblue', node_shape='s')

    # Draw node labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Draw edge labels with their constraints
    edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Program Visualization")
    plt.savefig(out_path)


def visualize_program_with_support(dataset, p_io_tt, p_io_tf, p_io_ft, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # First, group by data index
    index_2_ww_tt = defaultdict(list)
    for i, w, w2 in p_io_tt:
        index_2_ww_tt[i].append((w, w2))

    index_2_ww_tf = defaultdict(list)
    for i, w, w2 in p_io_tf:
        index_2_ww_tf[i].append((w, w2))

    index_2_ww_ft = defaultdict(list)
    for i, w, w2 in p_io_ft:
        index_2_ww_ft[i].append((w, w2))

    # Now, visualize each program
    for i in range(len(dataset)):
        if i not in index_2_ww_tt and i not in index_2_ww_tf and i not in index_2_ww_ft:
            continue
        if i not in index_2_ww_tt:
            continue
        img_fp = dataset[i].img_fp.replace(".jpg", ".png")
        # load image
        img = cv2.imread(img_fp)
        cnt = 0
        for w, w2 in index_2_ww_tt[i]:
            cnt += 1
            w_box = dataset[i]['boxes'][w]
            w2_box = dataset[i]['boxes'][w2]
            cv2.rectangle(img, (w_box[0], w_box[1]), (w_box[2], w_box[3]), (0, 255, 0), 2)
            cv2.rectangle(img, (w2_box[0], w2_box[1]), (w2_box[2], w2_box[3]), (0, 255, 0), 2)
            # line from first box to second box
            cv2.line(img, (w_box[0], w_box[1]), (w2_box[0], w2_box[1]), (0, 255, 0), 2)
        '''
        for w, w2 in index_2_ww_tf[i]:
            cnt += 1
            w_box = dataset[i]['boxes'][w]
            w2_box = dataset[i]['boxes'][w2]
            cv2.rectangle(img, (w_box[0], w_box[1]), (w_box[2], w_box[3]), (0, 0, 255), 2)
            cv2.rectangle(img, (w2_box[0], w2_box[1]), (w2_box[2], w2_box[3]), (0, 0, 255), 2)
            # line from first box to second box
            cv2.line(img, (w_box[0], w_box[1]), (w2_box[0], w2_box[1]), (0, 0, 255), 2)

        for w, w2 in index_2_ww_ft[i]:
            cnt += 1
            w_box = dataset[i]['boxes'][w]
            w2_box = dataset[i]['boxes'][w2]
            cv2.rectangle(img, (w_box[0], w_box[1]), (w_box[2], w_box[3]), (255, 0, 0), 2)
            cv2.rectangle(img, (w2_box[0], w2_box[1]), (w2_box[2], w2_box[3]), (255, 0, 0), 2)
            # line from first box to second box
            cv2.line(img, (w_box[0], w_box[1]), (w2_box[0], w2_box[1]), (255, 0, 0), 2)
        '''

        write_fp = os.path.join(out_dir, f'{i}.png')
        cv2.imwrite(write_fp, img)


if __name__ == '__main__':
    # Example Program
    program = "find((w0, w1, w2), (r0, r1), (rel(w0, r0, w1), rel(w1, r1, w2), (((w0.label == 'L_answer' and w1.label == 'L_answer') and w2.label == 'L_answer') and (r0.lbl == 'L_3' and r1.lbl == 'L_3')), w2)"

    visualize_program(program, 'example.png')

