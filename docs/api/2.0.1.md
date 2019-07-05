# API Docs - v2.0.1

!!! Info "Tested Siddhi Core version: *<a target="_blank" href="http://siddhi.io/en/v5.0/docs/query-guide/">5.0.2</a>*"
    It could also support other Siddhi Core minor versions.

## Tensorflow

### predict *<a target="_blank" href="http://siddhi.io/en/v5.0/docs/query-guide/#stream-processor">(Stream Processor)</a>*
<p style="word-wrap: break-word">Performs inferences (prediction) from an already built TensorFlow machine learning model. The types of models are unlimited (including image classifiers, deep learning models) as long as they satisfy the following conditions.<br>1. They are saved with the tag 'serve' in SavedModel format for more info see [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).<br>2. Model is initially trained and ready for inferences<br>3. Inference logic is written and saved in the model<br>4. signature_def is properly included in the metaGraphDef (a protocol buffer file which has information about the graph) and the key for prediction signature def is 'serving-default'<br><br>Also the prerequisites for inference are as follows.<br>1. User knows the names of the input and output nodes<br>2. Has a preprocessed data set of Java primitive types or their multidimensional arrays<br><br>Since each input is directly used to create a Tensor they should be of compatible shape and data type with the model.<br>The information related to input and output nodes can be retrieved from saved model signature def.signature_def can be read by using the saved_model_cli commands found at [https://www.tensorflow.org/programmers_guide/saved_model](https://www.tensorflow.org/programmers_guide/saved_model).<br>signature_def can be read in Python as follows<br><pre>
with tf.Session() as sess:
  md = tf.saved_model.loader.load(sess, ['serve'], export_dir)
  sig = md.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  print(sig)
</pre><br>Or you can read signature def from Java as follows,<br><pre>
final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default";
final SignatureDef sig =
      MetaGraphDef.parseFrom(model.metaGraphDef())
          .getSignatureDefOrThrow(DEFAULT_SERVING_SIGNATURE_DEF_KEY);
</pre><br>You will have to import the following in Java.<br><code>import org.tensorflow.framework.MetaGraphDef;</code><br><code>import org.tensorflow.framework.SignatureDef;</code></p>
<span id="syntax" class="md-typeset" style="display: block; font-weight: bold;">Syntax</span>

```
tensorFlow:predict(<STRING> absolute.path.to.model, <STRING> input.node.names, <STRING> output.node.names, <INT|STRING|DOUBLE|LONG|FLOAT|BOOL|OBJECT> attribute, <INT|STRING|DOUBLE|LONG|FLOAT|BOOL|OBJECT> ...)
tensorFlow:predict(<STRING> absolute.path.to.model, <INT|STRING|DOUBLE|LONG|FLOAT|BOOL|OBJECT> attribute, <INT|STRING|DOUBLE|LONG|FLOAT|BOOL|OBJECT> ...)
```

<span id="query-parameters" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">QUERY PARAMETERS</span>
<table>
    <tr>
        <th>Name</th>
        <th style="min-width: 20em">Description</th>
        <th>Default Value</th>
        <th>Possible Data Types</th>
        <th>Optional</th>
        <th>Dynamic</th>
    </tr>
    <tr>
        <td style="vertical-align: top">absolute.path.to.model</td>
        <td style="vertical-align: top; word-wrap: break-word">This is the absolute path to the model folder in the local machine.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">STRING</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">Yes</td>
    </tr>
    <tr>
        <td style="vertical-align: top">input.node.names</td>
        <td style="vertical-align: top; word-wrap: break-word">This is a variable length parameter. The names of the input nodes as comma separated strings.</td>
        <td style="vertical-align: top">-</td>
        <td style="vertical-align: top">STRING</td>
        <td style="vertical-align: top">Yes</td>
        <td style="vertical-align: top">Yes</td>
    </tr>
    <tr>
        <td style="vertical-align: top">output.node.names</td>
        <td style="vertical-align: top; word-wrap: break-word">This is a variable length parameter. The names of the output nodes as comma separated strings.</td>
        <td style="vertical-align: top">-</td>
        <td style="vertical-align: top">STRING</td>
        <td style="vertical-align: top">Yes</td>
        <td style="vertical-align: top">Yes</td>
    </tr>
    <tr>
        <td style="vertical-align: top">attribute</td>
        <td style="vertical-align: top; word-wrap: break-word">This is a variable length parameter. These are the attributes coming with events. Note that arrays should be cast to objects and sent.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">INT<br>STRING<br>DOUBLE<br>LONG<br>FLOAT<br>BOOL<br>OBJECT</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">Yes</td>
    </tr>
</table>
<span id="extra-return-attributes" class="md-typeset" style="display: block; font-weight: bold;">Extra Return Attributes</span>
<table>
    <tr>
        <th>Name</th>
        <th style="min-width: 20em">Description</th>
        <th>Possible Types</th>
    </tr>
    <tr>
        <td style="vertical-align: top">outputs</td>
        <td style="vertical-align: top; word-wrap: break-word">This is a variable length return attribute. The output tensors from the inference will be flattened out and sent in their primitive values. User is expected to know the shape of the output tensors if he/she wishes to reconstruct it. The shape and data type information can be retrieved from TensorFlow saved model signature_def. See the description of this extension for instructions on how to read signature_def</td>
        <td style="vertical-align: top">INT<br>STRING<br>DOUBLE<br>LONG<br>FLOAT<br>BOOL</td>
    </tr>
</table>

<span id="examples" class="md-typeset" style="display: block; font-weight: bold;">Examples</span>
<span id="example-1" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 1</span>
```
define stream InputStream (x Object, y Object);
@info(name = 'query1') 
from InputStream#tensorFlow:predict('home/MNIST', 'inputPoint', 'dropout', 'outputPoint', x, y) 
select outputPoint0, outputPoint1, outputPoint2, outputPoint3, outputPoint4, outputPoint5, outputPoint6, outputPoint7, outputPoint8, outputPoint9 
insert into OutputStream;

```
<p style="word-wrap: break-word">This is a query to get inferences from a MNIST model. This model takes in 2 inputs. One being the image as float array and other is keep probability array and sends out a Tensor with 10 elements. Our stream processor flattens the tensor and sends 10 floats each representing the probability of image being 0,1,...,9</p>

