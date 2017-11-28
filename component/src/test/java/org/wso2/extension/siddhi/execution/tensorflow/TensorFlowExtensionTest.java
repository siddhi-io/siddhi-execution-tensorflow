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

public class TensorFlowExtensionTest {
    private static final Logger logger = Logger.getLogger(TensorFlowExtensionTest.class);
    private volatile AtomicInteger count;

    @BeforeMethod
    public void init() {
        count = new AtomicInteger(0);
    }

    @Test
    public void initialTestingWithKMeansModel() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', 'outputPoint', x) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
                );

        SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);

        siddhiAppRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long l, Event[] events, Event[] events1) {
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
            inputHandler.send(new Object[]{"float:[1, -2]"});
            inputHandler.send(new Object[]{"float:[1, 2]"});
            inputHandler.send(new Object[]{"float:[5, 2]"});
        } catch (Exception e) {
            logger.error(e.getCause().getMessage());
        } finally {
            siddhiAppRuntime.shutdown();
        }
    }

    @Test
    public void initialTestingWithMNIST() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String, y String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/MNIST";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', " +
                        "'dropout', 'outputPoint', x, y) " +
                        "select outputPoint0, outputPoint1, outputPoint2, outputPoint3, outputPoint4, " +
                        "outputPoint5, outputPoint6, outputPoint7, outputPoint8, outputPoint9 " +
                        "insert into OutputStream;"
        );

        SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);

        siddhiAppRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long l, Event[] events, Event[] events1) {
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
            BufferedImage image = ImageIO.read(TensorFlowExtensionTest.class.getResource("/10.png"));
            float[] imgAsFloatArray = img2array(image);

            String imageAsString = "float:[";

            for (float num : imgAsFloatArray) {
                imageAsString  = imageAsString + num + ",";
            }

            imageAsString = imageAsString.substring(0, imageAsString.lastIndexOf(",") - 1) + "]";

            float[] keepProbArray = new float[1024];
            Arrays.fill(keepProbArray, 1f);

            String keepProbString = "float:[";

            for (float num : keepProbArray) {
                keepProbString  = keepProbString + num + ",";
            }

            keepProbString = keepProbString.substring(0, keepProbString.lastIndexOf(",") - 1) + "]";

            inputHandler.send(new Object[]{imageAsString, keepProbString});
        } catch (Exception e) {
            logger.error(e.getCause().getMessage());
        } finally {
            siddhiAppRuntime.shutdown();
        }
    }

    //pre-processing of the image to convert it to float array
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
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/Regression";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', " +
                        "'outputPoint', x) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
        );

        SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);

        siddhiAppRuntime.addCallback("query1", new QueryCallback() {
            @Override
            public void receive(long l, Event[] events, Event[] events1) {
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
            inputHandler.send(new Object[]{"double :[1,-2]"});
        } catch (Exception e) {
            logger.error(e.getCause().getMessage());
        } finally {
            siddhiAppRuntime.shutdown();
        }
    }

    @Test
    public void validatingFirstParamIsConstant() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String, y String);";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict(y, 'inputPoint', 'outputPoint', x) " +
                        "select outputPoint0, outputPoint1 " +
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
    public void validatingInputNameIsConstant() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String, y String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', y, 'outputPoint', x) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 2 is a input " +
                    "name which has to be a constant but found org.wso2.siddhi.core.executor" +
                    ".VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingOutputNameIsConstant() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String, y String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', y, x) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 3 is a output " +
                    "name which has to be a constant but found org.wso2.siddhi.core.executor" +
                    ".VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingFirstParamIsString() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String);";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict(2, 'inputPoint', 'outputPoint', x) " +
                        "select outputPoint0, outputPoint1 " +
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
    public void validatingInputNameIsString() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 5, 'outputPoint', x) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 2 is a input " +
                    "name which has to be a String but found INT"));
        }
    }

    @Test
    public void validatingOutputNameIsString() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', true, x) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 3 is a output " +
                    "name which has to be a String but found BOOL"));
        }
    }

    @Test
    public void validatingAttributeIsVariable() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', 'outputPoint', 4) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The parameter at index 4 is not a variable " +
                    "attribute (VariableExpressionExecutor) present in the stream definition. " +
                    "Found org.wso2.siddhi.core.executor.ConstantExpressionExecutor"));
        }
    }

    @Test
    public void validatingNumberOfInputsAndOutputsAreHonoured1() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', x) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("Invalid number of query parameters. Number " +
                    "of inputs and number of outputs are specified as 1 and 1 respectively. So the total number of " +
                    "query parameters should be 4 but 3 given."));
        }
    }

    @Test
    public void validatingNumberOfInputsAndOutputsAreHonoured2() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', x, x) " +
                        "select outputPoint0, outputPoint1 " +
                        "insert into OutputStream;"
        );

        try {
            SiddhiAppRuntime siddhiAppRuntime = siddhiManager.createSiddhiAppRuntime(inputStream + query);
        } catch (Exception e) {
            AssertJUnit.assertTrue(e instanceof SiddhiAppCreationException);
            AssertJUnit.assertTrue(e.getCause().getMessage().contains("The query parameter of index 3 is a output " +
                    "name which has to be a constant but found org.wso2.siddhi.core.executor" +
                    ".VariableExpressionExecutor"));
        }
    }

    @Test
    public void validatingInputNameIsPresentInGraphViaSignatureDef() throws Exception {
        SiddhiManager siddhiManager = new SiddhiManager();
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputX', 'outputPoint', x) " +
                        "select outputPoint0, outputPoint1 " +
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
        String inputStream = "define stream InputStream (x String);";

        String tempPath = TensorFlowExtensionTest.class.getResource("/10.png").getPath();
        String path = tempPath.substring(0, tempPath.lastIndexOf("/")) + "/TensorFlowModels/KMeans";

        String query = (
                "@info(name = 'query1') " +
                        "from InputStream#tensorFlow:predict('" + path + "', 'inputPoint', 'outputX', x) " +
                        "select outputX0, outputX9 " +
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
