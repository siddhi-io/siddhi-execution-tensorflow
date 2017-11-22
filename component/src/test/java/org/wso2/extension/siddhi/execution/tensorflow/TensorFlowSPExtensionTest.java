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
import org.testng.annotations.Test;
import org.wso2.siddhi.core.SiddhiAppRuntime;
import org.wso2.siddhi.core.SiddhiManager;
import org.wso2.siddhi.core.event.Event;
import org.wso2.siddhi.core.query.output.callback.QueryCallback;
import org.wso2.siddhi.core.stream.input.InputHandler;
import org.wso2.siddhi.core.util.EventPrinter;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import javax.imageio.ImageIO;

public class TensorFlowSPExtensionTest {
    private static final Logger logger = Logger.getLogger(TensorFlowSPExtensionTest.class);

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
                EventPrinter.print(events);
            }
        });

        siddhiAppRuntime.start();
        InputHandler inputHandler = siddhiAppRuntime.getInputHandler("InputStream");
        try {
            Object firstInput = new float[] {1, -2};
            inputHandler.send(new Object[]{firstInput});
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
                EventPrinter.print(events);
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
                EventPrinter.print(events);
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
}
