import java.util.List;

public class SingleLayerNetwork {

    double[] bias;
    double learningRate;
    double[][] weights;

    

    public static double[] textToFrequencyVector(String text) {
        double[] freq = new double[26];
        text = text.toLowerCase().replaceAll("[^a-z]", ""); 
        for (char c : text.toCharArray()) {
            int index = c - 'a';
            if (index >= 0 && index < 26) {
                freq[index]++;
            }
        }
        return freq;
    }

    public void train(List<Vector> data) {
        for (Vector vec : data) {
            double[] predicted = forward(vec.input);
            for (int j = 0; j < bias.length; j++) {
                double error = vec.target[j] - predicted[j];
                double delta = error * (predicted[j] * (1 - predicted[j])); //derivate
                bias[j] += learningRate * delta;
                for (int i = 0; i < vec.input.length; i++) {
                    weights[i][j] += learningRate * delta * vec.input[i];
                }
            }
        }
    }
}


public static double[] labelToOneHuge(String label) {
        double[] oneHuge = new double[4];
        switch (label.toLowerCase()) {
            case "english":
                oneHuge[0] = 1.0; break;
            case "german":
                oneHuge[1] = 1.0; break;
            case "polish":
                oneHuge[2] = 1.0; break;
            case "spanish":
                oneHuge[3] = 1.0; break;
        }
        return oneHuge;
    }

}
