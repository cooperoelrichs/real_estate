import os
import time

import tensorflow as tf


def train_and_evaluate(
    estimator, train_input_fn, eval_input_fn,
    training_steps, evaluation_steps, steps_between_evaluations,
    outputs_dir
):
    start_timestamp = time.time()
    current_step = 0

    while current_step < training_steps:
        next_checkpoint = min(
            current_step + steps_between_evaluations,
            training_steps
        )
        estimator.train(
            input_fn=train_input_fn,
            max_steps=next_checkpoint
        )
        current_step = next_checkpoint

        tf.logging.info('Starting to evaluate.')
        eval_results = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=evaluation_steps
        )
        tf.logging.info('Eval results: %s' % eval_results)

    elapsed_time = int(time.time() - start_timestamp)
    tf.logging.info(
        'Finished training up to step %d. Elapsed seconds %d.' %
        (training_steps, elapsed_time)
    )

    # tf.logging.info('Starting to export model.')
    # estimator.export_savedmodel(
    #     export_dir_base=os.path.join(outputs_dir, 'exports'),
    #     serving_input_receiver_fn=train_input_fn
    # )
