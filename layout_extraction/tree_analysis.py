from file_reader.layout.paragraph import *
from file_reader.layout.table import Cell, Table
from .tree import *


def post_process_node(nodes: List[Node], is_root: Callable) -> List[Node]:
    result: List[Node] = []
    for node in nodes:
        if node.box is not None and is_root(node.box):
            node.children = sort_box(node.children)
            result.append(node)
        else:
            if node.box is not None:
                result.append(Node(box=node.box))
            for child in node.children:
                result.append(Node(box=child))
    result = sort_box(result)
    return result


def group_one_column(containers: List[BoxContainer], is_root: Callable, unit_size=5, cells=None) -> List[Node]:
    """
    Group content into sections in the case of the resume with a single-column layout.
    :param containers: List of paragraphs in the resume.
    :param is_root:
    :param unit_size:
    :param cells:
    :return:
    """
    nodes = []
    rows = group_by_row(containers, is_same_row='is_intersection_y')
    node = Node()
    for row in rows:
        row = sort_box(row)
        root_id = -1
        for icol, r in enumerate(row):
            if is_root(r):
                root_id = icol
        if root_id >= 0:
            if node.has_info():
                nodes.append(node)
            node = Node(row[root_id])
            for r in row[root_id + 1:]:
                node.append_child(r)
        else:
            for r in row:
                node.append_child(r)
    if node.has_info():
        nodes.append(node)

    nodes = post_process_node(nodes, is_root)
    return nodes
