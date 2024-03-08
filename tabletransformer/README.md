This part of the repository is based on [Microsoft's Table Transformer](https://github.com/microsoft/table-transformer)

# Inference

Each image words is of the following format:
```json
{
    "words": {
        "id": "id"
        "bbox": [x0, y0, x1, y1]
        "text": "text"
    }
}
```
Or the following format:
```json
[
    {
        'span_num': <span num>,
        'line_num': <line num>,
        'block_num': <block_num>,
    }
]
```
