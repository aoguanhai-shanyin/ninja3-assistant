"""Build the card-matching ONNX graph without embedding card images."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from onnx import TensorProto, helper

from ninja3_assistant.paths import project_root


def _external_filters(name, card_count, height):
    shape = (card_count, 3, height, 110)
    tensor = TensorProto(name=name, dims=shape, data_type=TensorProto.FLOAT,
                         data_location=TensorProto.EXTERNAL)
    # The Worker supplies these tensors from installed card art at session creation.
    tensor.external_data.add(key='location', value=f'{name}.bin')
    return tensor


def make_model(card_count):
    node = helper.make_node
    float32 = TensorProto.FLOAT
    nodes = [
        node('ReduceMean', ['images'], ['image_means'], axes=[2, 3], keepdims=1),
        node('Sub', ['images', 'image_means'], ['centered']),
        node('Mul', ['centered', 'centered'], ['squared']),
    ]
    weights = [
        helper.make_tensor('channel_axis', TensorProto.INT64, [1], [1]),
        helper.make_tensor('epsilon', float32, [], [1e-4]),
        helper.make_tensor('minus_one', float32, [], [-1]),
        helper.make_tensor('plus_one', float32, [], [1]),
    ]
    outputs = []
    for prefix, height in (('', 65), ('upper_', 30)):
        kernel = [height, 110]
        weights.extend([
            _external_filters(prefix+'filters', card_count, height),
            helper.make_tensor(prefix+'pixels', float32, [], [height * 110]),
        ])
        # TM_CCOEFF_NORMED centers both sides per color channel. The filters
        # supplied by the Worker already contain normalized, centered pixels.
        nodes.extend([
            node('Conv', ['centered', prefix+'filters'], [prefix+'numerator'], kernel_shape=kernel),
            node('AveragePool', ['centered'], [prefix+'mean'], kernel_shape=kernel),
            node('AveragePool', ['squared'], [prefix+'square_mean'], kernel_shape=kernel),
            node('Mul', [prefix+'mean', prefix+'mean'], [prefix+'mean_squared']),
            node('Sub', [prefix+'square_mean', prefix+'mean_squared'], [prefix+'variance']),
            node('ReduceSum', [prefix+'variance', 'channel_axis'], [prefix+'total_variance'], keepdims=1),
            node('Mul', [prefix+'total_variance', prefix+'pixels'], [prefix+'variance_sum']),
            node('Max', [prefix+'variance_sum', 'epsilon'], [prefix+'positive_variance_sum']),
            node('Sqrt', [prefix+'positive_variance_sum'], [prefix+'denominator']),
            node('Div', [prefix+'numerator', prefix+'denominator'], [prefix+'correlations']),
            node('Clip', [prefix+'correlations', 'minus_one', 'plus_one'], [prefix+'bounded']),
            node('ReduceMax', [prefix+'bounded'], [prefix+'scores'], axes=[2, 3], keepdims=0),
        ])
        outputs.append(helper.make_tensor_value_info(prefix+'scores', float32, [14, card_count]))
    images = helper.make_tensor_value_info('images', float32, [14, 3, 85, 130])
    graph = helper.make_graph(nodes, 'Intelligence card portrait matching', [images], outputs, weights)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)


def main():
    assets = project_root()/'assets'
    catalog = json.loads((assets/'intelligence-cards.json').read_text(encoding='utf-8'))
    path = assets/'intelligence-portrait-batch.onnx'
    model = make_model(len(catalog['cards']))
    file = NamedTemporaryFile(dir=assets, prefix=path.name+'.', delete=False)
    temporary = Path(file.name)
    try:
        with file:
            file.write(model.SerializeToString())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print('Generated '+path.name)


if __name__ == '__main__':
    main()
