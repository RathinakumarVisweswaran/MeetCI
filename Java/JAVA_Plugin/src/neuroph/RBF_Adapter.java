package neuroph;

import interfaces.Adapter;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.RBFNetwork;
import org.neuroph.nnet.learning.RBFLearning;
import xml.MachineLearning;

import java.io.*;
import java.util.Arrays;

/**
 * Created by Rathinakumar on 7/13/2015.
 */
public class RBF_Adapter implements Adapter {

    @Override
    public String tranNeuralNetwork(MachineLearning machineLearning, String saveLocation) {


        // get the path to file with data
        String inputFileName = "data/Sine.txt";

        // create MultiLayerPerceptron neural network
        RBFNetwork neuralNet = new RBFNetwork(1, 15, 1);

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(inputFileName, 1, 1, ",", false);

        RBFLearning learningRule = ((RBFLearning)neuralNet.getLearningRule());
        learningRule.setLearningRate(0.02);
        learningRule.setMaxError(0.01);
        //learningRule.addListener(this);

        // train the network with training set
        neuralNet.learn(dataSet);

        System.out.println("Done training.");
        System.out.println("Testing network...");

        //testNeuralNetwork(neuralNet, dataSet);
        return null;
    }

    private String saveModel(RBFNetwork neuralNet, String saveLocation)
    {
        //saving the model
        File outputDir = new File(saveLocation+"\\"+System.currentTimeMillis());
        //File input = new File(inputFileName);
        if(outputDir.mkdir())
        {
            neuralNet.save(outputDir.getAbsolutePath().concat("\\MLP"));
            System.out.println("saving to " + outputDir.getAbsolutePath().concat("\\MLP"));
        }
        return outputDir.getAbsolutePath().concat("\\MLP");
    }

    public void testNeuralNetwork(String savedModel, String testDataFile, String output) throws IOException {
        File file = new File(savedModel);
        RBFNetwork model = (RBFNetwork) NeuralNetwork.load(new FileInputStream(file));
        DataSet testSet = DataSet.createFromFile(testDataFile, model.getInputsCount(), model.getOutputsCount(), ",", false);

        FileWriter outputFile = new FileWriter(new File(output));

        for(DataSetRow testSetRow : testSet.getRows()) {
            model.setInput(testSetRow.getInput());
            model.calculate();
            double[] networkOutput = model.getOutput();

            outputFile.write("Input: " + Arrays.toString(testSetRow.getInput()));
            outputFile.write(" Output: " + Arrays.toString(networkOutput));
        }
        outputFile.flush();
    }
}
