"""Build fixed RGB comparisons for DirectML; this is not model training."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build():
    nodes = [helper.make_node('Split', ['image'], ['r', 'g', 'b'], axis=1)]
    constants = {}

    def minimum(left, right, threshold, output):
        if right:
            nodes.append(helper.make_node('Sub', [left, right], [output+'_difference']))
            left = output+'_difference'
        key = 'constant_'+str(threshold)
        constants[key] = float(threshold)
        nodes.append(helper.make_node('GreaterOrEqual', [left, key], [output]))

    rules = {
        'blue': [('b', None, 225), ('g', None, 170), ('b', 'r', 30), ('g', 'r', 20)],
        'fire': [('r', None, 205), ('g', None, 120), ('r', 'b', 40)],
        'blackfire': [('r', None, 70), ('b', None, 110), ('b', 'g', 55), ('r', 'g', 35)],
    }
    for name, comparisons in rules.items():
        for i, (left, right, value) in enumerate(comparisons):
            minimum(left, right, value, f'{name}_{i}')
        combined = name+'_0'
        for i in range(1, len(comparisons)):
            output = f'{name}_and_{i}'
            nodes.append(helper.make_node('And', [combined, f'{name}_{i}'], [output]))
            combined = output
        nodes.append(helper.make_node('Cast', [combined], [name], to=TensorProto.FLOAT))
    initializers = [numpy_helper.from_array(np.array(value, np.float32), name)
                    for name, value in constants.items()]
    shape = [1, 1, 250, 640]
    graph = helper.make_graph(nodes, 'hayabusa-fixed-color-rules',
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, 250, 640])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in rules], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
    onnx.checker.check_model(model)
    path = Path(__file__).resolve().parents[1]/'assets/hayabusa-colors.onnx'
    onnx.save(model, path)
    return path


if __name__ == '__main__':
    print(build())
