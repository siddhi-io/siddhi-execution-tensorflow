# siddhi-execution-tensorflow
The **siddhi-execution-tensorflow** is an extension to <a target="_blank" href="https://wso2.github
.io/siddhi">Siddhi</a>  that adds support for inferences from pre-built TensorFlow SavedModels using Siddhi.

Find some useful links below:

* <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-tensorflow">Source code</a>
* <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-tensorflow/releases">Releases</a>
* <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-tensorflow/issues">Issue tracker</a>

## Latest API Docs

Latest API Docs is <a target="_blank" href="https://wso2-extensions.github.io/siddhi-execution-tensorflow/api/1.0.8">1.0.8</a>.

## How to use 

**Using the extension in <a target="_blank" href="https://github.com/wso2/product-sp">WSO2 Stream Processor</a>**

* You can use this extension in the latest <a target="_blank" href="https://github.com/wso2/product-sp/releases">WSO2 Stream Processor</a> that is a part of <a target="_blank" href="http://wso2.com/analytics?utm_source=gitanalytics&utm_campaign=gitanalytics_Jul17">WSO2 Analytics</a> offering, with editor, debugger and simulation support. 

**Using the extension as a <a target="_blank" href="https://wso2.github.io/siddhi/documentation/running-as-a-java-library">java library</a>**

* This extension can be added as a maven dependency along with other Siddhi dependencies to your project.

```
     <dependency>
        <groupId>org.wso2.extension.siddhi.execution.tensorflow</groupId>
        <artifactId>siddhi-execution-tensorflow</artifactId>
        <version>x.x.x</version>
     </dependency>
```

## Jenkins Build Status

---

|  Branch | Build Status |
| :------ |:------------ | 
| master  | [![Build Status](https://wso2.org/jenkins/view/All%20Builds/job/siddhi/job/siddhi-execution-tensorflow/badge/icon)](https://wso2.org/jenkins/view/All%20Builds/job/siddhi/job/siddhi-execution-tensorflow/) |

---

## Features

* <a target="_blank" href="https://wso2-extensions.github.io/siddhi-execution-tensorflow/api/1.0.8/#predict-stream-processor">predict</a> *<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>*<br><div style="padding-left: 1em;"><p>Performs inferences (prediction) from an already built TensorFlow machine learning model. The types of models are unlimited (including image classifiers, deep learning models) as long as they satisfy the following conditions.<br>1. They are saved with the tag 'serve' in SavedModel format (See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)<br>2. Model is initially trained and ready for inferences<br>3. Inference logic is written and saved in the model<br>4. signature_def is properly included in the metaGraphDef (a protocol buffer file which has information about the graph) and the key for prediction signature def is 'serving-default'<br><br>Also the prerequisites for inference are as follows.<br>1. User knows the names of the input and output nodes<br>2. Has a preprocessed data set of Java primitive types or their multidimensional arrays<br><br>Since each input is directly used to create a Tensor they should be of compatible shape and data type with the model.<br>The information related to input and output nodes can be retrieved from saved model signature def.signature_def can be read by using the saved_model_cli commands found at https://www.tensorflow.org/programmers_guide/saved_model<br>signature_def can be read in Python as follows<br>with tf.Session() as sess:<br>&nbsp;&nbsp;md = tf.saved_model.loader.load(sess, ['serve'], export_dir)<br>&nbsp;&nbsp;sig = md.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]<br>&nbsp;&nbsp;print(sig)<br><br>Or you can read signature def from Java as follows,<br>final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"; <br><br>final SignatureDef sig =<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MetaGraphDef.parseFrom(model.metaGraphDef())<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.getSignatureDefOrThrow(DEFAULT_SERVING_SIGNATURE_DEF_KEY);<br><br>You will have to import the following in Java.<br>import org.tensorflow.framework.MetaGraphDef;<br>import org.tensorflow.framework.SignatureDef;</p></div>

## How to Contribute
 
  * Please report issues at <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-tensorflow/issues">GitHub Issue Tracker</a>.
  
  * Send your contributions as pull requests to <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-tensorflow/tree/master">master branch</a>. 
 
## Contact us 

 * Post your questions with the <a target="_blank" href="http://stackoverflow.com/search?q=siddhi">"Siddhi"</a> tag in <a target="_blank" href="http://stackoverflow.com/search?q=siddhi">Stackoverflow</a>. 
 
 * Siddhi developers can be contacted via the mailing lists:
 
    Developers List   : [dev@wso2.org](mailto:dev@wso2.org)
    
    Architecture List : [architecture@wso2.org](mailto:architecture@wso2.org)
 
## Support 

* We are committed to ensuring support for this extension in production. Our unique approach ensures that all support leverages our open development methodology and is provided by the very same engineers who build the technology. 

* For more details and to take advantage of this unique opportunity contact us via <a target="_blank" href="http://wso2.com/support?utm_source=gitanalytics&utm_campaign=gitanalytics_Jul17">http://wso2.com/support/</a>. 
