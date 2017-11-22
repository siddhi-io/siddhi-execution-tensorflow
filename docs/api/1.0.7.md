# API Docs - v1.0.7

## Streamingml

### kMeansMiniBatch *<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>*

<p style="word-wrap: break-word">Performs K-Means clustering on a streaming data set. Data points can be of any dimension and the dimensionality is calculated from number of parameters. All data points to be processed in a single query should be of the same dimensionality. The Euclidean distance is taken as the distance metric. The algorithm resembles mini-batch K-Means. (refer Web-Scale K-Means Clustering by D.Sculley, Google, Inc.). </p>

<span id="syntax" class="md-typeset" style="display: block; font-weight: bold;">Syntax</span>
```
streamingml:kMeansMiniBatch(<INT> no.of.clusters, <DOUBLE> decay.rate, <INT> maximum.iterations, <INT> no.of.events.to.retrain, <DOUBLE|FLOAT|INT|LONG> model.features)
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
        <td style="vertical-align: top">no.of.clusters</td>
        <td style="vertical-align: top; word-wrap: break-word">The assumed number of natural clusters in the data set.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">INT</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">decay.rate</td>
        <td style="vertical-align: top; word-wrap: break-word">this is the decay rate of old data compared to new data. Value of this will be in [0,1]. 0 means only old data used and1 will mean that only new data is used</td>
        <td style="vertical-align: top">0.01</td>
        <td style="vertical-align: top">DOUBLE</td>
        <td style="vertical-align: top">Yes</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">maximum.iterations</td>
        <td style="vertical-align: top; word-wrap: break-word">Number of iterations, the process iterates until the number of maximum iterations is reached or the centroids do not change</td>
        <td style="vertical-align: top">50</td>
        <td style="vertical-align: top">INT</td>
        <td style="vertical-align: top">Yes</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">no.of.events.to.retrain</td>
        <td style="vertical-align: top; word-wrap: break-word">number of events to recalculate cluster centers. </td>
        <td style="vertical-align: top">20</td>
        <td style="vertical-align: top">INT</td>
        <td style="vertical-align: top">Yes</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">model.features</td>
        <td style="vertical-align: top; word-wrap: break-word">This is a variable length argument. Depending on the dimensionality of data points we will receive coordinates as features along each axis.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">DOUBLE<br>FLOAT<br>INT<br>LONG</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
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
        <td style="vertical-align: top">euclideanDistanceToClosestCentroid</td>
        <td style="vertical-align: top; word-wrap: break-word">Represents the Euclidean distance between the current data point and the closest centroid.</td>
        <td style="vertical-align: top">DOUBLE</td>
    </tr>
    <tr>
        <td style="vertical-align: top">closestCentroidCoordinate</td>
        <td style="vertical-align: top; word-wrap: break-word">This is a variable length attribute. Depending on the dimensionality(d) we will return closestCentroidCoordinate1 to closestCentroidCoordinated which are the d dimensional coordinates of the closest centroid from the model to the current event. This is the prediction result and this represents the cluster towhich the current event belongs to.</td>
        <td style="vertical-align: top">DOUBLE</td>
    </tr>
</table>

<span id="examples" class="md-typeset" style="display: block; font-weight: bold;">Examples</span>
<span id="example-1" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 1</span>
```
define stream InputStream (x double, y double);
@info(name = 'query1')
from InputStream#streamingml:kMeansMiniBatch(2, 0.2, 10, 20, x, y)
select closestCentroidCoordinate1, closestCentroidCoordinate2, x, y
insert into OutputStream;
```
<p style="word-wrap: break-word">This is an example where user gives all three hyper parameters. first 20 events will be consumed to build the model and from the 21st event prediction would start</p>

<span id="example-2" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 2</span>
```
define stream InputStream (x double, y double);
@info(name = 'query1')
from InputStream#streamingml:kMeansMiniBatch(2, x, y)
select closestCentroidCoordinate1, closestCentroidCoordinate2, x, y
insert into OutputStream;
```
<p style="word-wrap: break-word">This is an example where user has not specified hyper params. So default values will be used.</p>

### kMeansIncremental *<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>*

<p style="word-wrap: break-word">Performs K-Means clustering on a streaming data set. Data points can be of any dimension and the dimensionality is calculated from number of parameters. All data points to be processed by a query should be of the same dimensionality. The Euclidean distance is taken as the distance metric. The algorithm resembles Sequential K-Means Clustering at https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm </p>

<span id="syntax" class="md-typeset" style="display: block; font-weight: bold;">Syntax</span>
```
streamingml:kMeansIncremental(<INT> no.of.clusters, <DOUBLE> decay.rate, <DOUBLE|FLOAT|INT|LONG> model.features)
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
        <td style="vertical-align: top">no.of.clusters</td>
        <td style="vertical-align: top; word-wrap: break-word">The assumed number of natural clusters in the data set.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">INT</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">decay.rate</td>
        <td style="vertical-align: top; word-wrap: break-word">this is the decay rate of old data compared to new data. Value of this will be in [0,1]. 0 means only old data used and1 will mean that only new data is used</td>
        <td style="vertical-align: top">0.01</td>
        <td style="vertical-align: top">DOUBLE</td>
        <td style="vertical-align: top">Yes</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">model.features</td>
        <td style="vertical-align: top; word-wrap: break-word">This is a variable length argument. Depending on the dimensionality of data points we will receive coordinates as features along each axis.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">DOUBLE<br>FLOAT<br>INT<br>LONG</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
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
        <td style="vertical-align: top">euclideanDistanceToClosestCentroid</td>
        <td style="vertical-align: top; word-wrap: break-word">Represents the Euclidean distance between the current data point and the closest centroid.</td>
        <td style="vertical-align: top">DOUBLE</td>
    </tr>
    <tr>
        <td style="vertical-align: top">closestCentroidCoordinate</td>
        <td style="vertical-align: top; word-wrap: break-word">This is a variable length attribute. Depending on the dimensionality(D) we will return closestCentroidCoordinate1, closestCentroidCoordinate2,... closestCentroidCoordinateD which are the d dimensional coordinates of the closest centroid from the model to the current event. This is the prediction result and this represents the cluster to which the current event belongs to.</td>
        <td style="vertical-align: top">DOUBLE</td>
    </tr>
</table>

<span id="examples" class="md-typeset" style="display: block; font-weight: bold;">Examples</span>
<span id="example-1" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 1</span>
```
define stream InputStream (x double, y double);
@info(name = 'query1')
from InputStream#streamingml:kMeansIncremental(2, 0.2, x, y)
select closestCentroidCoordinate1, closestCentroidCoordinate2, x, y
insert into OutputStream;
```
<p style="word-wrap: break-word">This is an example where user provides the decay rate. First two events will be used to initiate the model since the required number of clusters is specified as 2. After the first event itself prediction would start.</p>

<span id="example-2" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 2</span>
```
define stream InputStream (x double, y double);
@info(name = 'query1')
from InputStream#streamingml:kMeansIncremental(2, x, y)
select closestCentroidCoordinate1, closestCentroidCoordinate2, x, y
insert into OutputStream;
```
<p style="word-wrap: break-word">This is an example where user doesnt give the decay rate so the default value will be used</p>

### updatePerceptronClassifier *<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>*

<p style="word-wrap: break-word">This extension builds/updates a linear binary classification Perceptron model.</p>

<span id="syntax" class="md-typeset" style="display: block; font-weight: bold;">Syntax</span>
```
streamingml:updatePerceptronClassifier(<STRING> model.name, <BOOL|STRING> model.label, <DOUBLE> learning.rate, <DOUBLE|INT> model.features)
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
        <td style="vertical-align: top">model.name</td>
        <td style="vertical-align: top; word-wrap: break-word">The name of the model to be built/updated.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">STRING</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">model.label</td>
        <td style="vertical-align: top; word-wrap: break-word">The attribute of the label or the class of the dataset.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">BOOL<br>STRING</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">learning.rate</td>
        <td style="vertical-align: top; word-wrap: break-word">The learning rate of the Perceptron algorithm.</td>
        <td style="vertical-align: top">0.1</td>
        <td style="vertical-align: top">DOUBLE</td>
        <td style="vertical-align: top">Yes</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">model.features</td>
        <td style="vertical-align: top; word-wrap: break-word">Features of the model that need to be attributes of the stream.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">DOUBLE<br>INT</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
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
        <td style="vertical-align: top">featureWeight</td>
        <td style="vertical-align: top; word-wrap: break-word">Weight of the &lt;feature.name&gt; of the model.</td>
        <td style="vertical-align: top">DOUBLE</td>
    </tr>
</table>

<span id="examples" class="md-typeset" style="display: block; font-weight: bold;">Examples</span>
<span id="example-1" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 1</span>
```
define stream StreamA (attribute_0 double, attribute_1 double, attribute_2 double, attribute_3 double, attribute_4 string );

from StreamA#streamingml:updatePerceptronClassifier('model1', attribute_4, 0.01, attribute_0, attribute_1, attribute_2, attribute_3) 
insert all events into outputStream;
```
<p style="word-wrap: break-word">This query builds/updates a Perceptron model named <code>model1</code> with a <code>0.01</code> learning rate using <code>attribute_0</code>, <code>attribute_1</code>, <code>attribute_2</code>, and <code>attribute_3</code> as features, and <code>attribute_4</code> as the label. Updated weights of the model are emitted to the OutputStream stream.</p>

<span id="example-2" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 2</span>
```
define stream StreamA (attribute_0 double, attribute_1 double, attribute_2 double,attribute_3 double, attribute_4 string );

 from StreamA#streamingml:updatePerceptronClassifier('model1', attribute_4, attribute_0, attribute_1, attribute_2, attribute_3) 
insert all events into outputStream;
```
<p style="word-wrap: break-word">This query builds/updates a Perceptron model named <code>model1</code> with a default <code>0.1</code> learning rate using <code>attribute_0</code>, <code>attribute_1</code>, <code>attribute_2</code>, and <code>attribute_3</code> as features, and <code>attribute_4</code> as the label. The updated weights of the model are appended to the outputStream.</p>

### perceptronClassifier *<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>*

<p style="word-wrap: break-word">This extension predicts using a linear binary classification Perceptron model.</p>

<span id="syntax" class="md-typeset" style="display: block; font-weight: bold;">Syntax</span>
```
streamingml:perceptronClassifier(<STRING> model.name, <DOUBLE> model.bias, <DOUBLE> model.threshold, <DOUBLE> model.features)
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
        <td style="vertical-align: top">model.name</td>
        <td style="vertical-align: top; word-wrap: break-word">The name of the model to be used.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">STRING</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">model.bias</td>
        <td style="vertical-align: top; word-wrap: break-word">The bias of the Perceptron algorithm.</td>
        <td style="vertical-align: top">0.0</td>
        <td style="vertical-align: top">DOUBLE</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">model.threshold</td>
        <td style="vertical-align: top; word-wrap: break-word">The threshold that separates the two classes. The value specified must be between zero and one.</td>
        <td style="vertical-align: top"> The output is a probability.</td>
        <td style="vertical-align: top">DOUBLE</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
    </tr>
    <tr>
        <td style="vertical-align: top">model.features</td>
        <td style="vertical-align: top; word-wrap: break-word">The features of the model that need to be attributes of the stream.</td>
        <td style="vertical-align: top"></td>
        <td style="vertical-align: top">DOUBLE</td>
        <td style="vertical-align: top">No</td>
        <td style="vertical-align: top">No</td>
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
        <td style="vertical-align: top">prediction</td>
        <td style="vertical-align: top; word-wrap: break-word">The predicted value (<code>true/false</code>)</td>
        <td style="vertical-align: top">BOOL</td>
    </tr>
    <tr>
        <td style="vertical-align: top">confidenceLevel</td>
        <td style="vertical-align: top; word-wrap: break-word">The probability of the prediction</td>
        <td style="vertical-align: top">DOUBLE</td>
    </tr>
</table>

<span id="examples" class="md-typeset" style="display: block; font-weight: bold;">Examples</span>
<span id="example-1" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 1</span>
```
define stream StreamA (attribute_0 double, attribute_1 double, attribute_2 double, attribute_3 double);

from StreamA#streamingml:perceptronClassifier('model1',0.0,0.5, attribute_0, attribute_1, attribute_2, attribute_3) 
insert all events into OutputStream;
```
<p style="word-wrap: break-word">This query uses a Perceptron model named <code>model1</code> with a <code>0.0</code> bias and a <code>0.5</code> threshold learning rate to predict the label of the feature vector represented by <code>attribute_0</code>, <code>attribute_1</code>, <code>attribute_2</code>, and <code>attribute_3</code>. The predicted label (<code>true/false</code>) is emitted to the <code>OutputStream</code> streamalong with the prediction confidence level(probability) and the feature vector. As a result, the OutputStream stream is defined as follows: (attribute_0 double, attribute_1 double, attribute_2 double, attribute_3 double, prediction bool, confidenceLevel double).</p>

<span id="example-2" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 2</span>
```
define stream StreamA (attribute_0 double, attribute_1 double, attribute_2 double, attribute_3 double);

from StreamA#streamingml:perceptronClassifier('model1',0.0, attribute_0, attribute_1, attribute_2, attribute_3) 
insert all events into OutputStream;
```
<p style="word-wrap: break-word">This query uses a Perceptron model named <code>model1</code> with a <code>0.0</code> bias to predict the label of the feature vector represented by <code>attribute_0</code>, <code>attribute_1</code>, <code>attribute_2</code>, and <code>attribute_3</code>. The prediction(<code>true/false</code>) is emitted to the <code>OutputStream</code>stream along with the prediction confidence level(probability) and the feature. As a result, the OutputStream stream is defined as follows: (attribute_0 double, attribute_1 double, attribute_2 double, attribute_3 double, prediction bool, confidenceLevel double).</p>

<span id="example-3" class="md-typeset" style="display: block; color: rgba(0, 0, 0, 0.54); font-size: 12.8px; font-weight: bold;">EXAMPLE 3</span>
```
define stream StreamA (attribute_0 double, attribute_1 double, attribute_2 double, attribute_3 double);

from StreamA#streamingml:perceptronClassifier(`model1`, attribute_0, attribute_1, attribute_2) 
insert all events into OutputStream;
```
<p style="word-wrap: break-word">This query uses a Perceptron model named <code>model1</code> with a default 0.0 bias to predict the label of the feature vector represented by <code>attribute_0</code>, <code>attribute_1</code>, and <code>attribute_2</code>. The predicted probability is emitted to the OutputStream stream along with the feature vector. As a result, the OutputStream is defined as follows: (attribute_0 double, attribute_1 double, attribute_2 double, attribute_3 double, prediction bool, confidenceLevel double).</p>

