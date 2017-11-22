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

import org.apache.log4j.Logger;
import org.testng.AssertJUnit;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;
import org.wso2.siddhi.core.SiddhiAppRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.exception.SiddhiAppCreationException;
import org.wso2.siddhi.core.query.output.callback.QueryCallback;
import org.wso2.siddhi.core.stream.input.InputHandler;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import javax.imageio.ImageIO;

public class TensorFlowSPExtensionTest {
    private static final Logger logger = Logger.getLogger(TensorFlowSPExtensionTest.class);
    private volatile AtomicInteger count;

    @BeforeMethod
    public void init() {
        count = new AtomicInteger(0);
    }

    @Test
    public void initialTestingWithKMeansModel() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
                );

        SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);

        siddhiAppRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long l, Event[] events, Event[] events1) {
//                EventPrinter.print(events);
                for (Event event: events) {
                    count.incrementAndGet();
                    switch (count.get()) {
                        case 1:
                            AssertJUnit.assertArrayEquals(new Float[]{0.49465084f, -0.29043096f}, new Object[]{
                                    event.getData(0), event.getData(1)});
                            break;
                        case 2:
                            AssertJUnit.assertArrayEquals(new Float[]{0.41562787f, 1.0673156f}, new Object[]{
                                    event.getData(0), event.getData(1)});
                            break;
                        case 3:
                            AssertJUnit.assertArrayEquals(new Float[]{2.4706075f, 0.86139846f}, new Object[]{
                                    event.getData(0), event.getData(1)});
                            break;
                    }
                }
            }
        });

        siddhiAppRuntime.start();
        InputHandler inputHandler = siddhiAppRuntime.getInputHandler("InputStream");
        try {
            Object firstInput = new float[] {1, -2};
            inputHandler.send(new Object[]{firstInput});
            Object secondInput = new float[] {1, 2};
            inputHandler.send(new Object[]{secondInput});
            Object thirdInput = new float[] {5, 2};
            inputHandler.send(new Object[]{thirdInput});
        } catch (Exception e) {
            logger.error(e.getCause().getMessage());
        } finally {
            siddhiAppRuntime.shutdown();
        }
    }

    @Test
    public void initialTestingWithMNIST() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object, y Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/MNIST";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 2, 1, 'input_tensor', " +
                        "'dropout/keep_prob', 'output_tensor', x, y) " +
                        "select output_tensor0, output_tensor1, output_tensor2, output_tensor3, output_tensor4, " +
                        "output_tensor5, output_tensor6, output_tensor7, output_tensor8, output_tensor9 " +
                        "insert into OutputStream;"
        );

        SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);

        siddhiAppRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long l, Event[] events, Event[] events1) {
//                EventPrinter.print(events);
                for (Event event: events) {
                    count.incrementAndGet();
                    switch (count.get()) {
                        case 1:
                            AssertJUnit.assertArrayEquals(new Float[]{1.0539598f, -2.2724361f, 2.1548953f, -0.71338075f,
                                    2.6006677f, 1.2193193f, 2.585527f, 2.5818956f, -0.32108462f, 0.8956634f},
                                    new Object[]{
                                    event.getData(0), event.getData(1), event.getData(2), event.getData(3),
                                            event.getData(4), event.getData(5), event.getData(6),
                                            event.getData(7), event.getData(8), event.getData(9)});
                            break;
                    }
                }
            }
        });

        siddhiAppRuntime.start();
        InputHandler inputHandler = siddhiAppRuntime.getInputHandler("InputStream");
        try {
            BufferedImage image = ImageIO.read(TensorFlowSPExtensionTest.class.getResource("/10.png"));
            float[] imgAsFloatArray = img2array(image);

            float[] keepProbArray = new float[1024];
            Arrays.fill(keepProbArray, 1f);

            inputHandler.send(new Object[]{imgAsFloatArray, keepProbArray});
        } catch (Exception e) {
            logger.error(e.getCause().getMessage());
        } finally {
            siddhiAppRuntime.shutdown();
        }
    }

    //preprocessing of the image to convert it to float array
    private static float[] img2array(BufferedImage bi) {
        float[] floatArray = new float[784];
        for (int i = 0; i < bi.getHeight(); i++) {
            for (int j = 0; j < bi.getWidth(); j++) {
                floatArray[i * 28 + j] = (bi.getRGB(i, j) & 0xFF) / 255.0f;
            }
        }
        return floatArray;
    }

    @Test
    public void initialTestingWithRegressionModel() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/Regression";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'x_coordinate', " +
                        "'y_coordinate', x) " +
                        "select y_coordinate0, y_coordinate1 " +
                        "insert into OutputStream;"
        );

        SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);

        siddhiAppRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long l, Event[] events, Event[] events1) {
//                EventPrinter.print(events);
                for (Event event: events) {
                    count.incrementAndGet();
                    switch (count.get()) {
                        case 1:
                            AssertJUnit.assertArrayEquals(new Double[]{5.0161824226379395, -4.023891925811768},
                                    new Object[]{event.getData(0), event.getData(1)});
                            break;
                    }
                }
            }
        });

        siddhiAppRuntime.start();
        InputHandler inputHandler = siddhiAppRuntime.getInputHandler("InputStream");
        try {
            Object firstInput = new double[] {1, -2};
            inputHandler.send(new Object[]{firstInput});
        } catch (Exception e) {
            logger.error(e.getCause().getMessage());
        } finally {
            siddhiAppRuntime.shutdown();
        }
    }

    @Test
    public void validatingFirstParamIsConstant() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object, y String);";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict(y, 1, 1, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );
        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("1st query parameter is the absolute path " +
                    "to model which has to be constant but found " +
                    "org.wso2.siddhi.core.executor.VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingSecondParamIsConstant() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object, y int);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', y, 1, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("2nd query parameter is number of inputs which " +
                    "has to be a constant but found org.wso2.siddhi.core.executor.VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingThirdParamIsConstant() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object, y int);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, y, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("3rd query parameter is number of outputs " +
                    "which has to be a constant but found org.wso2.siddhi.core.executor.VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingInputNameIsConstant() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object, y String);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, y, 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 4 is a input " +
                    "name which has to be a constant but found org.wso2.siddhi.core.executor" +
                    ".VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingOutputNameIsConstant() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object, y String);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'input', y, x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 5 is a output " +
                    "name which has to be a constant but found org.wso2.siddhi.core.executor" +
                    ".VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingFirstParamIsString() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict(2, 1, 1, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );
        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("1st query parameter is the absolute path to " +
                    "model which has to be of type String but found INT"));
        }
    }

    @Test
    public void validatingSecondParamIsInt() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1f, 1, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("2nd query parameter is number of inputs " +
                    "which has to be of type int but found FLOAT"));
        }
    }

    @Test
    public void validatingThirdParamIsInt() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 0.4, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("3rd query parameter is number of outputs " +
                    "which has to be of type int but found DOUBLE"));
        }
    }

    @Test
    public void validatingInputNameIsString() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 5, 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 4 is a input " +
                    "name which has to be a String but found INT"));
        }
    }

    @Test
    public void validatingOutputNameIsString() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'input', true, x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 5 is a output " +
                    "name which has to be a String but found BOOL"));
        }
    }

    @Test
    public void validatingAttributeIsVariable() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'input', 'output', 4) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The parameter at index 6 is not a variable " +
                    "attribute (VariableExpressionExecutor) present in the stream definition. " +
                    "Found org.wso2.siddhi.core.executor.ConstantExpressionExecutor"));
        }
    }

    @Test
    public void validatingRangeOfnoOfInputs() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', -1, 1, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("Number of inputs should be at least 1 but " +
                    "given as -1"));
        }
    }

    @Test
    public void validatingRangeOfnoOfOutputs() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, -3, 'input', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("Number of outputs should be at least 1 but " +
                    "given as -3"));
        }
    }

    @Test
    public void validatingNumberOfInputsAndOutputsAreHonoured1() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'input', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("Invalid number of query parameters. Number " +
                    "of inputs and number of outputs are specified as 1 and 1 respectively. So the total number of " +
                    "query parameters should be 6 but 5 given."));
        }
    }

    @Test
    public void validatingNumberOfInputsAndOutputsAreHonoured2() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'input', x, x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 5 is a output " +
                    "name which has to be a constant but found org.wso2.siddhi.core.executor" +
                    ".VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingInputNameIsPresentInGraphViaSignatureDef() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'inputX', 'output', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("inputX not present in the signature def. " +
                    "Please check the input node names"));
        }
    }

    @Test
    public void validatingOutputNameIsPresentInGraphViaSignatureDef() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x Object);";

        String tempPath = TensorFlowSPExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 1, 1, 'input', 'outputX', x) " +
                        "select output0, output1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("outputX not present in the signature def. " +
                    "Please check the output node names."));
        }
    }
}
