import tensorflow as tf

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:
            self.activation = tf.get_collection('activation')[0]
              # BY NAME:
            self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data})
