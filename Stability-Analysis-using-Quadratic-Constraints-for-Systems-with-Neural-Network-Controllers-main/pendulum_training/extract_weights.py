import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

chk_dir = 'data/4_layers_new_lqr_05-10-2024_01-14-45/21/'

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(chk_dir + 'model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(chk_dir))

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    for var in vars:
        var_name = var.name.replace('/', '_').replace(':', '_')
        if "policy" in var_name and "Adam" not in var_name:
            var_value = sess.run(var)
            if np.isscalar(var_value):
                np.savetxt(f"extracted_values/{var_name}.csv", [var_value], delimiter=',')
            else:
                np.savetxt(f"extracted_values/{var_name}.csv", var_value, delimiter=',')
