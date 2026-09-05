"""Produce a reference with TensorFlow CPU in a separate environment.

The legacy SavedModel contains optimizer objects that fail in TF2.20's object
loader. The supported TF1 graph loader restores its original inference graph
and variables. Directional finite differences avoid missing legacy graph
gradient registrations; no model/config bytecode is evaluated manually.
"""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saved-model", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rng = np.random.default_rng(2026)
    x = np.eye(4, dtype=np.float32)[rng.integers(4, size=(2, 2114))]
    directions = np.zeros((8, 2114, 4), np.float32)
    mid = np.repeat(x[:1], 8, axis=0)
    for i in range(8):
        pos = int(rng.integers(600, 1500))
        old = int(mid[i, pos].argmax())
        directions[i, pos, old], directions[i, pos, (old + 1) % 4] = -1, 1
    mid += .1 * directions
    epsilon = .001
    batch = np.concatenate([x, mid, mid + epsilon * directions, mid - epsilon * directions])
    with tf.Graph().as_default() as graph:
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
        with tf.compat.v1.Session(graph=graph, config=config) as session:
            meta = tf.compat.v1.saved_model.loader.load(session, ["serve"], args.saved_model)
            sig = meta.signature_def["serving_default"]
            inputs = {k: graph.get_tensor_by_name(v.name) for k, v in sig.inputs.items()}
            outputs = {k: graph.get_tensor_by_name(v.name) for k, v in sig.outputs.items()}
            y = session.run(outputs, feed_dict={inputs["sequence"]: batch,
                            inputs["profile_bias_input_0"]: np.zeros((len(batch), 1000, 2)),
                            inputs["counts_bias_input_0"]: np.zeros((len(batch), 2))})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=batch, profile=y["profile_predictions"],
                        counts=y["logcounts_predictions"], directions=directions, epsilon=epsilon)


if __name__ == "__main__":
    main()
