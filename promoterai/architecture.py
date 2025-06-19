from functools import partial
import tensorflow.keras.initializers as tki
import tensorflow.keras.layers as tkl
import tensorflow.keras as tk


_kernel_init = partial(tki.TruncatedNormal, stddev=0.01)


def _metaformer(model_dim, kernel_size, dilation_rate):
    def block(input_):
        layer = tkl.BatchNormalization(synchronized=True)(input_)
        layer = tkl.DepthwiseConv1D(
            kernel_size, dilation_rate=dilation_rate, padding='same'
        )(layer)
        intermediate = input_ + layer

        layer = tkl.BatchNormalization(synchronized=True)(intermediate)
        layer = tkl.Dense(
            model_dim * 4, activation='relu', kernel_initializer=_kernel_init()
        )(layer)
        layer = tkl.Dense(model_dim, kernel_initializer=_kernel_init())(layer)
        output_ = intermediate + layer
        return output_
    return block


def promoterAI(
        num_blocks,
        model_dim,
        output_dims,
        kernel_size=5,
        dilation_rate=lambda x: max(1, 2 ** (x // 2 - 1)),
        shortcut_layer_freq=4,
        output_crop=0
):
    input_ = tk.Input(shape=(None, 4))
    layers = list(range(num_blocks + 1))
    layers[0] = tkl.Conv1D(model_dim, 1, activation='relu')(input_)
    for i in range(num_blocks):
        layers[i + 1] = _metaformer(
            model_dim, kernel_size, dilation_rate(i)
        )(layers[i])

    outputs = [[] for _ in output_dims]
    for j, output_dim in enumerate(output_dims):
        outputs[j] = tkl.Average()([
            tkl.Dense(
                output_dim, activation='relu', name=f'output{j}_{i}'
            )(layers[i]) for i in range(num_blocks, 0, -shortcut_layer_freq)
        ])
        outputs[j] = tkl.Cropping1D(cropping=output_crop // 2)(outputs[j])
    return tk.Model(inputs=input_, outputs=tuple(outputs))


def twin_wrap(model):
    for layer in model.layers:
        layer.trainable = 'output0' in layer.name
    input_ref = tk.Input(shape=model.input_shape[1:])
    input_alt = tk.Input(shape=model.input_shape[1:])
    output_ref = _get_human_output(model(input_ref))
    output_alt = _get_human_output(model(input_alt))

    output_ = tkl.Subtract()([output_alt, output_ref])
    output_ = tkl.Lambda(lambda x: tk.backend.mean(x, axis=(1, 2)))(output_)
    return tk.Model(inputs=(input_ref, input_alt), outputs=output_)


# support for legacy models
def _get_human_output(outputs):
    if isinstance(outputs, tuple) or isinstance(outputs, list):
        return outputs[0]
    elif isinstance(outputs, dict):
        return outputs['human']
    else:
        return outputs
