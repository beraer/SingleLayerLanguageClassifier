package linearNeuron;

import java.util.List;

public class SingleLayerNetwork {

    double[] biases; //[numOfOutputs]
    double learningRate;
    double[][] weights; //[numOfOutputs][inputSize]


    public SingleLayerNetwork(int inputSize, int outputSize, double learningRate) {
        biases = new double[outputSize]; //character amount eg.26
        this.learningRate = learningRate;
        weights = new double[outputSize][inputSize]; //outputSize is for expected class number

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = Math.random() * 0.01 - 0.001;
            }
            biases[i] = Math.random() * 0.01 - 0.001;
        }
    }

    public double[] computeOutput(double[] inputVector){
        double[] outputVector = new double[weights.length]; //one output per row(local rep)
        // z = SUM(WiXi + b)
        for (int i = 0; i < weights.length; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputVector.length; j++) {
                sum += weights[i][j] * inputVector[j];
            }
            outputVector[i] = sum; //linear
        }
        return outputVector;
    }


    public double[] derivative(double[] output, double[] target) {
        // L = MSE = 1/2(predicted - actual)^2
        // derivative of MSE will be exactly predicted - actual
        double[] gradient = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[i] = (output[i] - target[i]);
        }
        return gradient;
    }


    public void train(List<Vector> trainingData, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Vector v : trainingData) {
                double[] input = v.getInput();
                double[] target = v.getTarget();
                double[] output = computeOutput(input);

                double[] gradient = derivative(output, target);

                //update weights
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[i].length; j++) {
                        // dL/dw_ij = gradient[i] * input[j]
                        weights[i][j] -= learningRate * gradient[i] * input[j];
                    }

                    biases[i] -= learningRate * gradient[i];
                }
            }
        }
    }

    public int predict(double[] input) {
        double[] output = computeOutput(input);
        int bestIndex = 0;
        double bestVal = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > bestVal) {
                bestVal = output[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    public double evaluateAccuracy(List<Vector> testData) {
        int correct = 0;
        for (Vector v : testData) {
            int pred = predict(v.getInput());
            int actual = argmax(v.getTarget());
            if (pred == actual) {
                correct++;
            }
        }
        return 100.0 * correct / testData.size();
    }

    public void evaluateMetrics(List<Vector> testData) {
        int numClasses = weights.length;
        int[][] confusionMatrix = new int[numClasses][numClasses];

        for (Vector v : testData) {
            int actual = argmax(v.getTarget());
            int predicted = predict(v.getInput());
            confusionMatrix[actual][predicted]++;
        }

        System.out.println("Confusion Matrix:");
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                System.out.print(confusionMatrix[i][j] + "\t");
            }
            System.out.println();
        }

        double[] precision = new double[numClasses];
        double[] recall = new double[numClasses];
        double[] f1 = new double[numClasses];

        for (int i = 0; i < numClasses; i++) {
            int tp = confusionMatrix[i][i];
            int fp = 0;
            int fn = 0;
            for (int j = 0; j < numClasses; j++) {
                if (j != i) {
                    fp += confusionMatrix[j][i]; //same column different rows
                    fn += confusionMatrix[i][j]; //same row different columns
                }
            }
            precision[i] = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0; //precision = tp / tp + fp
            recall[i]    = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0; //recall = tp / tp + fn
            f1[i]        = (precision[i] + recall[i]) > 0 ? 2 * precision[i] * recall[i] / (precision[i] + recall[i]) : 0; //f-m = 2PR/(P+R)
        }

        String[] classNames = {"English", "Spanish", "German", "Polish"};
        double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
        System.out.println("Metrics per class (in %):");
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%s: Precision = %.2f, Recall = %.2f, F1 = %.2f%n",
                    classNames[i], precision[i] * 100, recall[i] * 100, f1[i] * 100);
            totalPrecision += precision[i];
            totalRecall += recall[i];
            totalF1 += f1[i];
        }
        System.out.printf("Macro-Averaged: Precision = %.2f, Recall = %.2f, F1 = %.2f%n",
                (totalPrecision / numClasses) * 100, (totalRecall / numClasses) * 100, (totalF1 / numClasses) * 100);
    }

    private int argmax(double[] arr) {
        int bestIndex = 0;
        double bestVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > bestVal) {
                bestVal = arr[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    public static double[] textToFrequencyVector(String text) {
        double[] freq = new double[26];
        for(int i = 0; i < text.length(); i++) {
            freq[text.charAt(i) - 'a']++;
        }

        //normalize
        double sum = 0.0;
        for(int i = 0; i < freq.length; i++) {
            sum += freq[i];
        }
        for(int i = 0; i < freq.length; i++) {
            freq[i] /= sum;
        }
        return freq;
    }
}
