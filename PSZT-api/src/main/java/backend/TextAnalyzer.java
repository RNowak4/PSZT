package backend;

import org.encog.neural.networks.BasicNetwork;

import java.util.Map;

public interface TextAnalyzer extends Analyzer, Training {
    Map<String, Integer> getCategories();

    void loadCategoriesFromFile(final String fileName);

    void setNeuralNetwork(final BasicNetwork network);
}
