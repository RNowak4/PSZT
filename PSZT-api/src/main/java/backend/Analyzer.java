package backend;

import org.encog.ml.data.MLData;
import org.encog.neural.data.NeuralData;
import org.encog.neural.networks.BasicNetwork;

public interface Analyzer {
    MLData analyzeFile(final String fileName);

    MLData analyzeText(final String text);
}
