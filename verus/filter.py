import re
from typing import Any


class Node:
    def evaluate(self, tags: dict[str, float]) -> bool:
        raise NotImplementedError()


class StringNode(Node):
    def __init__(self, value: str):
        self.value = value

    def evaluate(self, tags: dict[str, float]) -> bool:
        return self.value in tags


class RegexNode(Node):
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def evaluate(self, tags: dict[str, float]) -> bool:
        return any(self.pattern.search(tag) for tag in tags)


class OrNode(Node):
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    def evaluate(self, tags: dict[str, float]) -> bool:
        return any(node.evaluate(tags) for node in self.nodes)


class AndNode(Node):
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    def evaluate(self, tags: dict[str, float]) -> bool:
        return all(node.evaluate(tags) for node in self.nodes)


def parse_node(data: Any) -> Node:
    if isinstance(data, str):
        return StringNode(data)
    elif isinstance(data, dict):
        if data["type"] == "regex":
            return RegexNode(data["value"])
        elif data["type"] == "or":
            return OrNode([parse_node(item) for item in data["value"]])
        elif data["type"] == "and":
            return AndNode([parse_node(item) for item in data["value"]])
    elif isinstance(data, list):
        return AndNode([parse_node(item) for item in data])
    raise ValueError("Unsupported node type")
