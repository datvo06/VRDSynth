from typing import *


class SectionGrouping:

    def group_to_tree(self, entities: List[Dict]) -> List[Dict]:
        sections = self.group_entities(entities, is_header=lambda x: x["is_header"])
        for section in sections:
            section["content"] = self.group_entities(section["content"], is_header=lambda x: x["label"] == "HEADER")
        return sections

    def group_entities(self, entities: List[Dict], is_header: Callable) -> List[Dict]:
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
