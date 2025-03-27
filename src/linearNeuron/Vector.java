package linearNeuron;

public class Vector {
    double[] input;
    double[] target;

    public Vector(double[] input, double[] target) {
        this.input = input;
        this.target = target;
    }

    public double[] getInput() {
        return input;
    }

    public double[] getTarget() {
        return target;
    }
}
