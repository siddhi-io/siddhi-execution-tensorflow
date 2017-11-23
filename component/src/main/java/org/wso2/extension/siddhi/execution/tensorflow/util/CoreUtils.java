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

package org.wso2.extension.siddhi.execution.tensorflow.util;

import org.tensorflow.DataType;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.framework.SignatureDef;
import org.wso2.siddhi.core.executor.ExpressionExecutor;
import org.wso2.siddhi.core.executor.VariableExpressionExecutor;
import org.wso2.siddhi.query.api.definition.Attribute;
import org.wso2.siddhi.query.api.exception.SiddhiAppValidationException;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Core util functions for TensorFlow SP extension
 */
public class CoreUtils {

    public static VariableExpressionExecutor[] extractAndValidateTensorFlowInputs(
            ExpressionExecutor[] attributeExpressionExecutors, int startIndex, int noOfInputs) {

        VariableExpressionExecutor[] inputVariableExpressionExecutors = new VariableExpressionExecutor[noOfInputs];

        for (int i = startIndex; i < startIndex + noOfInputs; i++) {
            if (attributeExpressionExecutors[i] instanceof  VariableExpressionExecutor) {
                inputVariableExpressionExecutors[i - startIndex] = ((VariableExpressionExecutor)
                        attributeExpressionExecutors[i]);
            } else {
                throw new SiddhiAppValidationException("The parameter at index " + (i + 1) + " is not a variable " +
                        "attribute (VariableExpressionExecutor) present in the stream definition. Found " +
                        attributeExpressionExecutors[i].getClass().getCanonicalName());
            }
        }
        return inputVariableExpressionExecutors;
    }

    public static List<Attribute> getReturnAttributeList(SignatureDef signatureDef, int noOfOutputs,
                                                         SavedModelBundle tensorFlowSavedModel,
                                                         String[] outputNamesArray) {
        List<Attribute> attributeList = new ArrayList<>(noOfOutputs);
        for (int i = 0; i < noOfOutputs; i++) {

            String nodeName = signatureDef.getOutputsMap().get(outputNamesArray[i]).getName();
            String opName = nodeName.substring(0, nodeName.lastIndexOf(":"));

            DataType outputDataType = tensorFlowSavedModel.graph().operation(opName).output(0)
                    .dataType();
            if (outputDataType == DataType.STRING) {
                attributeList.add(new Attribute(outputNamesArray[i], Attribute.Type.STRING));
            } else {

                Shape outputShape = tensorFlowSavedModel.graph().operation(opName).output(0).shape();
                //Finding the total number of elements
                int numElements = 1;

                for (int k = 0; k < outputShape.numDimensions(); k++) {
                    if (outputShape.size(k) == -1) {
                        continue;
                    }
                    numElements *= outputShape.size(k);
                }

                for (int j = 0; j < numElements; j++) {
                    if (outputDataType == DataType.FLOAT) {
                        attributeList.add(new Attribute(outputNamesArray[i] + j, Attribute.Type.FLOAT));

                    } else if (outputDataType == DataType.BOOL) {
                        attributeList.add(new Attribute(outputNamesArray[i] + j, Attribute.Type.BOOL));

                    } else if (outputDataType == DataType.DOUBLE) {
                        attributeList.add(new Attribute(outputNamesArray[i] + j, Attribute.Type.DOUBLE));

                    } else if (outputDataType == DataType.INT32 || outputDataType == DataType.UINT8) {
                        attributeList.add(new Attribute(outputNamesArray[i] + j, Attribute.Type.INT));

                    } else if (outputDataType == DataType.INT64) {
                        attributeList.add(new Attribute(outputNamesArray[i] + j, Attribute.Type.LONG));
                    }
                }
            }
        }
        return attributeList;
    }

    public static Object[] getOutputObjectArray(List<Tensor> outputTensorList) {
        List<Object> objectList = new LinkedList<>();

        for (Tensor outputTensor : outputTensorList) {

            DataType tensorDataType = outputTensor.dataType();

            if (tensorDataType == DataType.FLOAT) {
                FloatBuffer floatBuffer = FloatBuffer.allocate(outputTensor.numElements());
                outputTensor.writeTo(floatBuffer);
                float[] floatArray = floatBuffer.array();

                for (float value : floatArray) {
                    objectList.add(value);
                }

            } else if (tensorDataType == DataType.DOUBLE) {
                DoubleBuffer doubleBuffer = DoubleBuffer.allocate(outputTensor.numElements());
                outputTensor.writeTo(doubleBuffer);
                double[] doubleArray = doubleBuffer.array();

                for (double value : doubleArray) {
                    objectList.add(value);
                }

            } else if (tensorDataType == DataType.INT32) {
                IntBuffer intBuffer = IntBuffer.allocate(outputTensor.numElements());
                outputTensor.writeTo(intBuffer);
                int[] intArray = intBuffer.array();

                for (int value : intArray) {
                    objectList.add(value);
                }

            } else if (tensorDataType == DataType.INT64) {
                LongBuffer longBuffer = LongBuffer.allocate(outputTensor.numElements());
                outputTensor.writeTo(longBuffer);
                long[] longArray = longBuffer.array();

                for (long value : longArray) {
                    objectList.add(value);
                }

            } else {
                ByteBuffer byteBuffer = ByteBuffer.allocate(outputTensor.numBytes());
                outputTensor.writeTo(byteBuffer);
                byte[] byteArray = byteBuffer.array();

                if (tensorDataType == DataType.STRING) {
                    String recoveredString = new String(byteArray, StandardCharsets.UTF_8);
                    objectList.add(recoveredString);

                } else if (tensorDataType == DataType.UINT8) {
                    for (byte value : byteArray) {
                        objectList.add((int) value);
                    }
                } else {
                    for (byte value : byteArray) {
                        if (value == 1) {
                            objectList.add(true);
                        } else {
                            objectList.add(false);
                        }
                    }
                }
            }

            outputTensor.close();
        }
        Object[] outputs = new Object[objectList.size()];
        objectList.toArray(outputs);
        return outputs;
    }

}
