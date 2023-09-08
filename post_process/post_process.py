from typing import *


def _x0(entity):
    if isinstance(entity, list):
        return min(map(_x0, entity))
    return entity['x0']


def _x1(entity):
    if isinstance(entity, list):
        return max(map(_x1, entity))
    return entity['x1']


def _y0(entity):
    if isinstance(entity, list):
        return min(map(_y0, entity))
    return entity['y0']


def _y1(entity):
    if isinstance(entity, list):
        return max(map(_y1, entity))
    return entity['y1']


def _height(entity):
    return _y1(entity) - _y0(entity)


def is_same_row(entity1, entity2) -> bool:
    if _y0(entity1) > _y1(entity2):
        return False
    if _y0(entity2) > _y1(entity1):
        return False

    return True


def is_same_col(entity1, entity2) -> bool:
    if _x0(entity1) > _x1(entity2):
        return False
    if _x0(entity2) > _x1(entity1):
        return False

    return True


class PostProcess:

    def merge_text(self, entities: List[Dict]) -> str:
        entities = sorted(entities, key=lambda entity: (int(entity["y0"] / 2), int(entity["x0"])))
        return " ".join([entity["text"] for entity in entities])

    def process_key_value(self, entities: List[Dict]) -> Dict:
        others = []
        keys = []
        values = []
        for entity in entities:
            if entity["label"] == "QUESTION":
                keys.append(entity)
            elif entity["label"] == "ANSWER":
                values.append(entity)
            else:
                others.append(entity)

        # Find the value to the right of the key
        for value in values:
            for key in keys:
                if key.get("value"):
                    continue
                if is_same_row(key, value) and key["x1"] < value["x0"] < key["x1"] + 2 * _height(key):
                    value["key"] = key
                    key["value"] = value
                    break

        # Find the value below the key
        for value in values:
            if value.get("key"):
                continue

            for key in keys:
                if key.get("value"):
                    continue
            above_keys = [key for key in keys if not key.get("value") and _y1(key) < _y0(value)]
            if above_keys:
                key = max(above_keys, key=_y0)
                value["key"] = key
                if "values" in key:
                    key["values"].append(value)
                else:
                    key["values"] = [value]
        pairs = []
        for value in values:
            if value.get("key"):
                pairs.append({"key": value["key"]["text"], "value": value["text"]})
            else:
                others.append(value)

        for key in keys:
            if not key.get("value") and not key.get("values"):
                pairs.append({"key": key["text"], "value": ""})

        content = self.merge_text(others)
        return {
            "description": content,
            "key-value": pairs
        }

    def process_section(self, section: Dict) -> Dict:
        title = self.merge_text(section["title"])

        return {
            "title": title,
            "content": self.process_key_value(section["content"])
        }

    def process(self, groups: List[Dict]) -> List[Dict]:
        result = []
        for group in groups:
            if group["title"]:
                result.append({
                    "header": self.merge_text(group["title"]),
                    "sections": [self.process_section(section) for section in group["content"]]
                })
        return result
