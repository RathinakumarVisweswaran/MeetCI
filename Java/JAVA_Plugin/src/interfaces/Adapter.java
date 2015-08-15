package interfaces;

import org.neuroph.core.transfer.TransferFunction;
import xml.MachineLearning;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Rathinakumar on 7/11/2015.
 */
public interface Adapter {
    public String tranNeuralNetwork(MachineLearning machineLearning, String saveLocation) throws IOException;
    public void testNeuralNetwork(String savedModel, String testDataFile, String output) throws IOException;
}
