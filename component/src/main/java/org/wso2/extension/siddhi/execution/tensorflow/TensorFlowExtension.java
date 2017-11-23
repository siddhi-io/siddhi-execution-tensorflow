/*
 * Copyright (c) 2017, WSO2 Inc. (http://www.wso2.org) All Rights Reserved.
 *
 * WSO2 Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.wso2.extension.siddhi.execution.tensorflow;

import com.google.protobuf.InvalidProtocolBufferException;
import org.apache.log4j.Logger;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.wso2.extension.siddhi.execution.tensorflow.util.CoreUtils;
import org.wso2.siddhi.annotation.Example;
import org.wso2.siddhi.annotation.Extension;
import org.wso2.siddhi.annotation.Parameter;
import org.wso2.siddhi.annotation.ReturnAttribute;
import org.wso2.siddhi.annotation.util.DataType;
import org.wso2.siddhi.core.config.SiddhiAppContext;
import org.wso2.siddhi.core.event.ComplexEventChunk;
import org.wso2.siddhi.core.event.stream.StreamEvent;
import org.wso2.siddhi.core.event.stream.StreamEventCloner;
import org.wso2.siddhi.core.event.stream.populater.ComplexEventPopulater;
import org.wso2.siddhi.core.exception.SiddhiAppCreationException;
import org.wso2.siddhi.core.executor.ConstantExpressionExecutor;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.query.processor.stream.StreamProcessor;
import org.wso2.siddhi.core.util.config.ConfigReader;
import org.wso2.siddhi.query.api.definition.AbstractDefinition;
import org.wso2.siddhi.query.api.definition.Attribute;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.wso2.extension.siddhi.execution.tensorflow.util.CoreUtils.getOutputObjectArray;
import static org.wso2.extension.siddhi.execution.tensorflow.util.CoreUtils.getReturnAttributeList;

/***
 * Stream processor extension to support inferences from TensorFlow models
 */
@Extension(
        name = "predict",
        namespace = "tensorFlow",
        description = "Performs inferences (prediction) from an already built TensorFlow machine learning model. " +
                "The types of models are unlimited (including image classifiers, deep learning models) as long as " +
                "they satisfy the following conditions.\n" +
                "1. They are saved with the tag 'serve' in SavedModel format\n" +
                "2. Model is initially trained and ready for inferences\n" +
                "3. Inference logic is written and saved in the model\n" +
                "4. signature_def is properly included in the metaGraphDef (a protocol buffer file which has " +
                "information about the graph) and the key for prediction signature " +
                "def is 'serving-default'\n" +
                "\n" +
                "Also the prerequisites for inference are as follows.\n" +
                "1. User knows the names of the input and output nodes\n" +
                "2. Has a preprocessed data set of Java primitive types or their multidimensional arrays\n" +
                "\n" +
                "Since each input is directly used to create a Tensor they should be of compatible shape and " +
                "data type with the model.\n" +
                "The information related to input and output nodes can be retrieved from saved model signature def." +
                "Signature def can be read in Python as follows\n" +
                "with tf.Session() as sess:\n" +
                "  md = tf.saved_model.loader.load(sess, ['serve'], export_dir)\n" +
                "  sig = md.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]\n" +
                "  print(sig)\n" +
                "\n" +
                "Or you can read signature def from Java as follows,\n" +
                "final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = \"serving_default\"; \n" +
                "\n" +
                "final SignatureDef sig =\n" +
                "      MetaGraphDef.parseFrom(model.metaGraphDef())\n" +
                "          .getSignatureDefOrThrow(DEFAULT_SERVING_SIGNATURE_DEF_KEY);\n" +
                "\n" +
                "You will have to import the following in Java.\n" +
                "import org.tensorflow.framework.MetaGraphDef;\n" +
                "import org.tensorflow.framework.SignatureDef;",
        parameters = {
                @Parameter(
                        name = "absolute.path.to.model",
                        description = "This is the absolute path to the model folder in the local machine.",
                        type = {DataType.STRING}
                ),
                @Parameter(
                        name = "input.node.names",
                        description = "This is a variable length parameter. The names of the input nodes as comma " +
                                "separated strings.",
                        type = {DataType.STRING}
                ),
                @Parameter(
                        name = "output.node.names",
                        description = "This is a variable length parameter. The names of the output nodes as comma " +
                                "separated strings.",
                        type = {DataType.STRING}
                ),
                @Parameter(
                        name = "attributes",
                        description = "This is a variable length parameter. These are the attributes coming with " +
                                "events. Note that arrays should be cast to objects and sent.",
                        type = {DataType.INT, DataType.STRING, DataType.DOUBLE, DataType.LONG, DataType.FLOAT,
                                DataType.BOOL, DataType.OBJECT}
                ),
        },
        returnAttributes = {
                @ReturnAttribute(
                        name = "outputs",
                        description = "This is a variable length return attribute. The output tensors from the " +
                                "inference will be flattened out and sent in their primitive values. User is " +
                                "expected to know the shape of the output tensors if he/she wishes to reconstruct " +
                                "it. The shape and data type information can be retrieved from TensorFlow saved " +
                                "model signature_def. See the description of this extension for instructions on how " +
                                "to read signature_def",
                        type = {DataType.INT, DataType.STRING, DataType.DOUBLE, DataType.LONG, DataType.FLOAT,
                                DataType.BOOL}
                ),
        },
        examples = {
            @Example(
                    syntax = "define stream InputStream (x Object, y Object);\n" +
                            "@info(name = 'query1') \n" +
                            "from InputStream#tensorFlow:predict('home/MNIST', 'inputPoint', " +
                            "'dropout', 'outputPoint', x, y) \n" +
                            "select outputPoint0, outputPoint1, outputPoint2, outputPoint3, outputPoint4, " +
                            "outputPoint5, outputPoint6, outputPoint7, outputPoint8, outputPoint9 \n" +
                            "insert into OutputStream;\n",
                    description = "This is a query to get inferences from a MNIST model. This model takes in 2 " +
                            "inputs. One being the image as float array and other is keep probability array and " +
                            "sends out a Tensor with 10 elements. Our stream processor flattens the tensor and sends " +
                            "10 floats each representing the probability of image being 0,1,...,9"
            )
        }
)
public class TensorFlowExtension extends StreamProcessor {
    private static final Logger logger = Logger.getLogger(TensorFlowExtension.class);
    private String[] inputVariableNamesArray;
    private String[] outputVariableNamesArray;
    private int noOfInputs;
    private int noOfOutputs;
    private VariableExpressionExecutor[] inputVariableExpressionExecutors;
    private Session tensorFlowSession;
    private SignatureDef signatureDef;
    private SavedModelBundle tensorFlowSavedModel;

    @Override
    protected List<Attribute> init(AbstractDefinition abstractDefinition, ExpressionExecutor[] expressionExecutors,
                                   ConfigReader configReader, SiddhiAppContext siddhiAppContext) {

        final int minConstantParams = 3;

        //Checking if at least minimum number of constant params are present in the query
        if (attributeExpressionLength < minConstantParams) {
            String message = "Insufficient number of parameters. Query should have at least 5 constant parameters " +
                    "and appropriate number of variable parameters but given " + attributeExpressionLength;
            logger.error(siddhiAppContext.getName() + message);
            throw new SiddhiAppCreationException(message);
        }

        //expressionExecutors[0] --> absolute path to model
        String modelPath;
        if (!(attributeExpressionExecutors[0] instanceof ConstantExpressionExecutor)) {
            throw new SiddhiAppCreationException("1st query parameter is the absolute path to model which has to be " +
                    "constant but found " + this.attributeExpressionExecutors[0].getClass().getCanonicalName());
        }
        if (attributeExpressionExecutors[0].getReturnType() == Attribute.Type.STRING) {
            modelPath = (String) ((ConstantExpressionExecutor) attributeExpressionExecutors[0]).getValue();
        } else {
            throw new SiddhiAppCreationException("1st query parameter is the absolute path to model which has to " +
                    "be of type String but found " + attributeExpressionExecutors[0].getReturnType());
        }

        //loading the saved model
        final String servingTag = "serve";
        tensorFlowSavedModel = SavedModelBundle.load(modelPath, servingTag);
        tensorFlowSession = tensorFlowSavedModel.session();

        //Loading signatureDef
        final String defaultServingSignatureDefKey = "serving_default";
        try {
            signatureDef =
                    MetaGraphDef.parseFrom(tensorFlowSavedModel.metaGraphDef())
                            .getSignatureDefOrThrow(defaultServingSignatureDefKey);
        } catch (InvalidProtocolBufferException e) {
            throw new SiddhiAppCreationException("Error while reading signature def." + e.getMessage(), e);
        }

        noOfInputs = signatureDef.getInputsCount();
        noOfOutputs = signatureDef.getOutputsCount();

        //Checking if the specified number of inputs are given
        final int noOfQueryParams = 1 + 2 * noOfInputs + noOfOutputs;
        if (attributeExpressionLength != noOfQueryParams) {
            String message = "Invalid number of query parameters. Number of inputs and number of outputs are " +
                    "specified as " + noOfInputs + " and " + noOfOutputs + " respectively. So the total number of " +
                    "query parameters should be " + noOfQueryParams + " but " + attributeExpressionLength + " given.";
            logger.error(siddhiAppContext.getName() + message);
            throw new SiddhiAppCreationException(message);
        }

        //Instantiate inputVariableNamesArray
        inputVariableNamesArray = new String[noOfInputs];

        //Instantiate outputVariableNamesArray
        outputVariableNamesArray = new String[noOfOutputs];

        //Validating and extracting the input names from the query parameters
        for (int i = 0; i < noOfInputs; i++) {
            int index = i + 1;
            if (!(attributeExpressionExecutors[index] instanceof ConstantExpressionExecutor)) {
                throw new SiddhiAppCreationException("The query parameter of index " + (index + 1) + " is a input " +
                        "name which has to be a constant but found " +
                        this.attributeExpressionExecutors[index].getClass().getCanonicalName());
            }
            if (attributeExpressionExecutors[index].getReturnType() == Attribute.Type.STRING) {
                inputVariableNamesArray[i] = (String) ((ConstantExpressionExecutor)
                        attributeExpressionExecutors[index]).getValue();
            } else {
                throw new SiddhiAppCreationException("The query parameter of index " + (index + 1) + " is a input " +
                        "name which has to be a String but found " +
                        this.attributeExpressionExecutors[index].getReturnType());
            }
        }

        //Validating and extracting the output names from the query parameters
        for (int i = 0; i < noOfOutputs; i++) {
            int index = i + 1 + noOfInputs;
            if (!(attributeExpressionExecutors[index] instanceof ConstantExpressionExecutor)) {
                throw new SiddhiAppCreationException("The query parameter of index " + (index + 1) + " is a output " +
                        "name which has to be a constant but found " +
                        this.attributeExpressionExecutors[index].getClass().getCanonicalName());
            }
            if (attributeExpressionExecutors[index].getReturnType() == Attribute.Type.STRING) {
                outputVariableNamesArray[i] = (String) ((ConstantExpressionExecutor)
                        attributeExpressionExecutors[index]).getValue();
            } else {
                throw new SiddhiAppCreationException("The query parameter of index " + (index + 1) + " is a output " +
                        "name which has to be a String but found " +
                        this.attributeExpressionExecutors[index].getReturnType());
            }
        }

        //Checking whether the node names are present in the signature def
        for (String inputNodeName : inputVariableNamesArray) {
            if (!(signatureDef.getInputsMap().containsKey(inputNodeName))) {
                throw new SiddhiAppCreationException(inputNodeName + " not present in the signature def. Please " +
                        "check the input node names");
            }
        }

        for (String outputNodeName: outputVariableNamesArray) {
            if (!(signatureDef.getOutputsMap().containsKey(outputNodeName))) {
                throw new SiddhiAppCreationException(outputNodeName + " not present in the signature def. Please " +
                        "check the output node names.");
            }
        }

        int inputValuesStartIndex = 1 + noOfInputs + noOfOutputs;

        //Extracting and validating variable expression executors
        inputVariableExpressionExecutors = CoreUtils.extractAndValidateTensorFlowInputs(attributeExpressionExecutors,
                inputValuesStartIndex, noOfInputs);

        return getReturnAttributeList(signatureDef, noOfOutputs, tensorFlowSavedModel, outputVariableNamesArray);
    }

    @Override
    protected void process(ComplexEventChunk<StreamEvent> complexEventChunk, Processor processor,
                           StreamEventCloner streamEventCloner, ComplexEventPopulater complexEventPopulater) {
        while (complexEventChunk.hasNext()) {
            StreamEvent streamEvent = complexEventChunk.next();

            Session.Runner tensorFlowRunner = tensorFlowSession.runner();
            //getting TensorFlow input values from stream event and feeding the model
            for (int i = 0; i < noOfInputs; i++) {
                try {
                    Tensor input = Tensor.create(inputVariableExpressionExecutors[i].execute(streamEvent));
                    tensorFlowRunner = tensorFlowRunner.feed(
                            signatureDef.getInputsMap().get(inputVariableNamesArray[i]).getName(), input);
                } catch (Throwable e) {
                    //catching throwable and logging because we don't want to stop the app if one bad input is given
                    logger.error("Error while feeding input " + inputVariableNamesArray[i] + ". " + e.getMessage());
                }
            }

            //fetching all the required outputs
            for (int i = 0; i < noOfOutputs; i++) {
                tensorFlowRunner = tensorFlowRunner.fetch(
                        signatureDef.getOutputsMap().get(outputVariableNamesArray[i]).getName());
            }

            //Running the session and getting the output tensors
            List<Tensor> outputTensors = tensorFlowRunner.run();

            complexEventPopulater.populateComplexEvent(streamEvent, getOutputObjectArray(outputTensors));
        }
        nextProcessor.process(complexEventChunk);
    }

    @Override
    public void start() {
        tensorFlowSession = tensorFlowSavedModel.session();
    }

    @Override
    public void stop() {
        //If the model learns with predictions then we need to persist the model and restore.
        //But current TensorFlow Java API r1.4 doesn't support serving of models
        tensorFlowSession.close();
    }

    @Override
    public Map<String, Object> currentState() {
        Map<String, Object> map = new HashMap();
        map.put("savedModel", tensorFlowSavedModel);
        return map;
    }

    @Override
    public void restoreState(Map<String, Object> map) {
        tensorFlowSavedModel = (SavedModelBundle) map.get("savedModel");
    }
}
