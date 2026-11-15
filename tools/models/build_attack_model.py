"""Build the red-attack color segmentation model used by the battle prototype."""
from ninja3_assistant.paths import project_root

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


constants = {
    'scale': 255.0, 'red_hue': 1 / 3, 'pink_hue': 8 / 30,
    'strong_saturation': 150.0, 'strong_value': 130.0,
    'weak_saturation': 60.0, 'weak_value': 150.0,
}
initializers = [numpy_helper.from_array(np.array(value, dtype=np.float32), name)
                for name, value in constants.items()]
nodes = [
    helper.make_node('Split', ['image'], ['r', 'g', 'b'], axis=1),
    helper.make_node('Min', ['g', 'b'], ['minimum']),
    helper.make_node('Sub', ['r', 'minimum'], ['delta']),
    helper.make_node('Sub', ['g', 'b'], ['gb']),
    helper.make_node('Sub', ['b', 'g'], ['bg']),
    helper.make_node('Mul', ['delta', 'red_hue'], ['red_limit']),
    helper.make_node('Mul', ['delta', 'pink_hue'], ['pink_limit']),
    helper.make_node('GreaterOrEqual', ['r', 'g'], ['r_ge_g']),
    helper.make_node('GreaterOrEqual', ['r', 'b'], ['r_ge_b']),
    helper.make_node('LessOrEqual', ['gb', 'red_limit'], ['hue_red']),
    helper.make_node('LessOrEqual', ['bg', 'pink_limit'], ['hue_pink']),
    helper.make_node('And', ['r_ge_g', 'r_ge_b'], ['red_max']),
    helper.make_node('And', ['hue_red', 'hue_pink'], ['hue_in_range']),
    helper.make_node('And', ['red_max', 'hue_in_range'], ['red_hue_mask']),
    helper.make_node('Mul', ['delta', 'scale'], ['saturation_scaled']),
]
for name in ('strong', 'weak'):
    nodes.extend([
        helper.make_node('Mul', ['r', name + '_saturation'], [name + '_sat_limit']),
        helper.make_node('GreaterOrEqual', ['saturation_scaled', name + '_sat_limit'], [name + '_sat']),
        helper.make_node('GreaterOrEqual', ['r', name + '_value'], [name + '_bright']),
        helper.make_node('And', [name + '_sat', name + '_bright'], [name + '_sat_bright']),
        helper.make_node('And', ['red_hue_mask', name + '_sat_bright'], [name + '_bool']),
        helper.make_node('Cast', [name + '_bool'], [name], to=TensorProto.FLOAT),
    ])
graph = helper.make_graph(nodes, 'boss-red-segmentation',
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, 300, 590])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 300, 590])
         for name in ('strong', 'weak')], initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
onnx.checker.check_model(model)
onnx.save(model, project_root() / 'assets/boss-red-segmentation.onnx')
