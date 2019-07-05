Siddhi Execution TensorFlow
======================================

  [![Jenkins Build Status](https://wso2.org/jenkins/job/siddhi/job/siddhi-execution-tensorflow/badge/icon)](https://wso2.org/jenkins/job/siddhi/job/siddhi-execution-tensorflow/)
  [![GitHub (pre-)Release](https://img.shields.io/github/release/siddhi-io/siddhi-execution-tensorflow/all.svg)](https://github.com/siddhi-io/siddhi-execution-tensorflow/releases)
  [![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/siddhi-io/siddhi-execution-tensorflow.svg)](https://github.com/siddhi-io/siddhi-execution-tensorflow/releases)
  [![GitHub Open Issues](https://img.shields.io/github/issues-raw/siddhi-io/siddhi-execution-tensorflow.svg)](https://github.com/siddhi-io/siddhi-execution-tensorflow/issues)
  [![GitHub Last Commit](https://img.shields.io/github/last-commit/siddhi-io/siddhi-execution-tensorflow.svg)](https://github.com/siddhi-io/siddhi-execution-tensorflow/commits/master)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The **siddhi-execution-tensorflow extension** is a <a target="_blank" href="https://siddhi.io/">Siddhi</a> extension that provides support for running pre-built TensorFlow models. 

For information on <a target="_blank" href="https://siddhi.io/">Siddhi</a> and it's features refer <a target="_blank" href="https://siddhi.io/redirect/docs.html">Siddhi Documentation</a>. 

## Download

* Versions 2.x and above with group id `io.siddhi.extension.*` from <a target="_blank" href="https://mvnrepository.com/artifact/io.siddhi.extension.execution.tensorflow/siddhi-execution-tensorflow/">here</a>.
* Versions 1.x and lower with group id `org.wso2.extension.siddhi.*` from <a target="_blank" href="https://mvnrepository.com/artifact/org.wso2.extension.siddhi.execution.tensorflow/siddhi-execution-tensorflow">here</a>.

## Latest API Docs 

Latest API Docs is <a target="_blank" href="https://siddhi-io.github.io/siddhi-execution-tensorflow/api/2.0.0">2.0.0</a>.

## Features

* <a target="_blank" href="https://siddhi-io.github.io/siddhi-execution-tensorflow/api/2.0.0/#predict-stream-processor">predict</a> *(<a target="_blank" href="http://siddhi.io/en/v5.0/docs/query-guide/#stream-processor">Stream Processor</a>)*<br> <div style="padding-left: 1em;"><p>Performs inferences (prediction) from an already built TensorFlow machine learning model. The types of models are unlimited (including image classifiers, deep learning models) as long as they satisfy the following conditions.<br>1. They are saved with the tag 'serve' in SavedModel format for more info see [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).<br>2. Model is initially trained and ready for inferences<br>3. Inference logic is written and saved in the model<br>4. signature_def is properly included in the metaGraphDef (a protocol buffer file which has information about the graph) and the key for prediction signature def is 'serving-default'<br><br>Also the prerequisites for inference are as follows.<br>1. User knows the names of the input and output nodes<br>2. Has a preprocessed data set of Java primitive types or their multidimensional arrays<br><br>Since each input is directly used to create a Tensor they should be of compatible shape and data type with the model.<br>The information related to input and output nodes can be retrieved from saved model signature def.signature_def can be read by using the saved_model_cli commands found at [https://www.tensorflow.org/programmers_guide/saved_model](https://www.tensorflow.org/programmers_guide/saved_model).<br>signature_def can be read in Python as follows<br><pre>with tf.Session() as sess:
  md = tf.saved_model.loader.load(sess, ['serve'], export_dir)
  sig = md.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  print(sig)
</pre><br>Or you can read signature def from Java as follows,<br><pre>final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"; 

final SignatureDef sig =
      MetaGraphDef.parseFrom(model.metaGraphDef())
          .getSignatureDefOrThrow(DEFAULT_SERVING_SIGNATURE_DEF_KEY);
</pre><br>You will have to import the following in Java.<br><code>import org.tensorflow.framework.MetaGraphDef;</code><br><code>import org.tensorflow.framework.SignatureDef;</code></p></div>

## Dependencies 

There are no other dependencies needed for this extension. 

## Installation

For installing this extension on various siddhi execution environments refer Siddhi documentation section on <a target="_blank" href="https://siddhi.io/redirect/add-extensions.html">adding extensions</a>.

## Support and Contribution

* We encourage users to ask questions and get support via <a target="_blank" href="https://stackoverflow.com/questions/tagged/siddhi">StackOverflow</a>, make sure to add the `siddhi` tag to the issue for better response.

* If you find any issues related to the extension please report them on <a target="_blank" href="https://github.com/siddhi-io/siddhi-execution-tensorflow/issues">the issue tracker</a>.

* For production support and other contribution related information refer <a target="_blank" href="https://siddhi.io/community/">Siddhi Community</a> documentation.
