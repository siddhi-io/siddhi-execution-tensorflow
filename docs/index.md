# siddhi-execution-streamingml
The **siddhi-execution-streamingml** is an extension to <a target="_blank" href="https://wso2.github
.io/siddhi">Siddhi</a>  that performs streaming machine learning on event streams.

Find some useful links below:
* <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-streamingml">Source code</a>
* <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-streamingml/releases">Releases</a>
* <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-streamingml/issues">Issue tracker</a>

## Latest API Docs

Latest API Docs is <a target="_blank" href="https://wso2-extensions.github.io/siddhi-execution-streamingml/api/1.0.8">1.0.8</a>.

## How to use 

**Using the extension in <a target="_blank" href="https://github.com/wso2/product-sp">WSO2 Stream Processor</a>**

* You can use this extension in the latest <a target="_blank" href="https://github.com/wso2/product-sp/releases">WSO2 Stream Processor</a> that is a part of <a target="_blank" href="http://wso2.com/analytics?utm_source=gitanalytics&utm_campaign=gitanalytics_Jul17">WSO2 Analytics</a> offering, with editor, debugger and simulation support. 

* This extension is shipped by default with WSO2 Stream Processor, if you wish to use an alternative version of this extension you can replace the component <a target="_blank" href="https://https://github.com/wso2-extensions/siddhi-execution-streamingml/releases">jar</a> that can be found in the `<STREAM_PROCESSOR_HOME>/lib` directory.

**Using the extension as a <a target="_blank" href="https://wso2.github.io/siddhi/documentation/running-as-a-java-library">java library</a>**

* This extension can be added as a maven dependency along with other Siddhi dependencies to your project.

```
     <dependency>
        <groupId>org.wso2.extension.siddhi.execution.streamingml</groupId>
        <artifactId>siddhi-execution-streamingml</artifactId>
        <version>x.x.x</version>
     </dependency>
```

## Features

* <a target="_blank" href="https://wso2-extensions.github.io/siddhi-execution-streamingml/api/1.0.8/#kmeansincremental-stream-processor">kMeansIncremental</a> *(<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>)*<br><div style="padding-left: 1em;"><p>Performs K-Means clustering on a streaming data set. Data points can be of any dimension and the dimensionality is calculated from number of parameters. All data points to be processed by a query should be of the same dimensionality. The Euclidean distance is taken as the distance metric. The algorithm resembles Sequential K-Means Clustering at https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm </p></div>
* <a target="_blank" href="https://wso2-extensions.github.io/siddhi-execution-streamingml/api/1.0.8/#kmeansminibatch-stream-processor">kMeansMiniBatch</a> *(<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>)*<br><div style="padding-left: 1em;"><p>Performs K-Means clustering on a streaming data set. Data points can be of any dimension and the dimensionality is calculated from number of parameters. All data points to be processed in a single query should be of the same dimensionality. The Euclidean distance is taken as the distance metric. The algorithm resembles mini-batch K-Means. (refer Web-Scale K-Means Clustering by D.Sculley, Google, Inc.). </p></div>
* <a target="_blank" href="https://wso2-extensions.github.io/siddhi-execution-streamingml/api/1.0.8/#perceptronclassifier-stream-processor">perceptronClassifier</a> *(<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>)*<br><div style="padding-left: 1em;"><p>This extension predicts using a linear binary classification Perceptron model.</p></div>
* <a target="_blank" href="https://wso2-extensions.github.io/siddhi-execution-streamingml/api/1.0.8/#updateperceptronclassifier-stream-processor">updatePerceptronClassifier</a> *(<a target="_blank" href="https://wso2.github.io/siddhi/documentation/siddhi-4.0/#stream-processor">(Stream Processor)</a>)*<br><div style="padding-left: 1em;"><p>This extension builds/updates a linear binary classification Perceptron model.</p></div>

## How to Contribute
 
  * Please report issues at <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-streamingml/issues">GitHub Issue Tracker</a>.
  
  * Send your contributions as pull requests to <a target="_blank" href="https://github.com/wso2-extensions/siddhi-execution-streamingml/tree/master">master branch</a>. 
 
## Contact us 

 * Post your questions with the <a target="_blank" href="http://stackoverflow.com/search?q=siddhi">"Siddhi"</a> tag in <a target="_blank" href="http://stackoverflow.com/search?q=siddhi">Stackoverflow</a>. 
 
 * Siddhi developers can be contacted via the mailing lists:
 
    Developers List   : [dev@wso2.org](mailto:dev@wso2.org)
    
    Architecture List : [architecture@wso2.org](mailto:architecture@wso2.org)
 
## Support 

* We are committed to ensuring support for this extension in production. Our unique approach ensures that all support leverages our open development methodology and is provided by the very same engineers who build the technology. 

* For more details and to take advantage of this unique opportunity contact us via <a target="_blank" href="http://wso2.com/support?utm_source=gitanalytics&utm_campaign=gitanalytics_Jul17">http://wso2.com/support/</a>. 
