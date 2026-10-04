import json

def parse_row(row):
    """
    Convert one JevBench row into a Python datastructure (it is served as json string)
    """

    state = json.loads(row['state'])
    question = json.loads(row['question'])

    return{
        "id": row['id'],
        "source": row["source"],
        "primitive": row["primitive"],
        "state": state,
        "question": question,
        "label": parse_label(
            row['primitive'],
            row['label'],
        ),
    }


def parse_label(primitive, label):
    """
    Convert dataset label into a Python type
    """

    if primitive == "choice":
        return label
    
    if primitive == "noul":
        return int(label)
    
    if primitive == "score":
        return int(label)
    
    raise ValueError(
        f"Unknown primitive: {primitive}"
    )