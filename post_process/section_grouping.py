from typing import *
from layout_extraction.funsd_utils import Entity
from layout_extraction.layout_extraction import HEADER_LABEL


def is_section_header(entity: Entity, width: int) -> bool:
    if (entity["label"] == HEADER_LABEL) and 0.4 * width < 0.5 * (entity["x0"] + entity["x1"]) < 0.6 * width:
        return True
    return False


class SectionGrouping:

    def group_to_tree(self, entities: List[Entity], width: int) -> List[Dict]:
        sections = self.group_entities(entities, is_header=lambda x: is_section_header(x, width))
        for section in sections:
            section["content"] = self.group_entities(section["content"], is_header=lambda x: x.label == HEADER_LABEL)
        return sections

    def group_entities(self, entities: List[Entity], is_header: Callable) -> List[Dict[str, List[Entity]]]:
        entities = sorted(entities, key=lambda entity: entity["y0"])
        groups = []
        group = {"content": [], "title": []}
        for entity in entities:
            if is_header(entity):
                if group["content"]:
                    groups.append(group)
                    group = {"content": [], "title": [entity]}
                else:
                    group["title"].append(entity)
            else:
                group["content"].append(entity)
        if group["content"] or group["title"]:
            groups.append(group)
        return groups
