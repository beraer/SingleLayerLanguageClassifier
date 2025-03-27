package linearNeuron;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        int inputSize = 26;
        int outputSize = 4;
        double learningRate = 0.01;
        int epochs = 500;

        SingleLayerNetwork net = new SingleLayerNetwork(inputSize, outputSize, learningRate);

        String trainPath = "data/lang.train.csv";
        List<Vector> trainingData = CSVReader.readCSV(trainPath);

        System.out.println("Training on " + trainingData.size() + " samples...");
        net.train(trainingData, epochs);
        System.out.println("Training completed.");

        String testPath = "data/lang.test.csv";
        List<Vector> testData = CSVReader.readCSV(testPath);

        double accuracy = net.evaluateAccuracy(testData);
        System.out.printf("Accuracy on test set: %.2f%%%n", accuracy);

        net.evaluateMetrics(testData);
    }
}
