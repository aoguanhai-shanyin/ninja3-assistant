"""Create the fixed S badge matching model from the bundled template."""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

root = Path(__file__).resolve().parents[1]
template = np.asarray(Image.open(root / 'assets/rank-s.png').convert('L'), dtype=np.float32)[:45] / 255.0
weight = np.stack((np.ones_like(template), -2 * template), axis=0)[None, ...].astype(np.float32)
offset = np.array([np.sum(template * template)], dtype=np.float32)
nodes = [helper.make_node('Mul', ['image', 'image'], ['squared']),
         helper.make_node('Concat', ['squared', 'image'], ['features'], axis=1),
         helper.make_node('Conv', ['features', 'kernel'], ['correlation']),
         helper.make_node('Add', ['correlation', 'offset'], ['distance']),
         helper.make_node('ReduceMin', ['distance'], ['best'], keepdims=0)]
graph = helper.make_graph(nodes, 's-rank-template',
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 1, 75, 110])],
        [helper.make_tensor_value_info('best', TensorProto.FLOAT, [])],
        [numpy_helper.from_array(weight, 'kernel'), numpy_helper.from_array(offset, 'offset')])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
onnx.checker.check_model(model)
onnx.save(model, root / 'assets/rank-s.onnx')
