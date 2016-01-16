package backend;

import org.encog.neural.data.NeuralData;
import org.encog.neural.networks.BasicNetwork;

public interface Analyzer {
    NeuralData analyzeFile(final String fileName);

    NeuralData analyzeText(final String text);

    void setNeuralNetwork(final BasicNetwork network);
}
