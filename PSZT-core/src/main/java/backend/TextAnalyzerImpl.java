package backend;

import org.encog.neural.data.NeuralData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class TextAnalyzerImpl implements TextAnalyzer {
    private Analyzer analyzer;
    private BasicNetwork network;
    private Training training;
    private Map<String, Integer> categories;

    public TextAnalyzerImpl(double desiredError, int maxEpochs,
                            final Class<? extends Train> trainingMethodType) {
        this.categories = new HashMap<>();
        this.network = new BasicNetwork();
        this.training = new TrainingImpl(desiredError, maxEpochs, network, trainingMethodType);
        this.training.setCategories(categories);
        // TODO analyzer init...
        // analyzer.setNeuralNetwork(network);
    }

    @Override
    public Map<String, Integer> getCategories() {
        return categories;
    }

    @Override
    public NeuralData analyzeFile(final String fileName) {
        return analyzer.analyzeFile(fileName);
    }

    @Override
    public NeuralData analyzeText(final String text) {
        return analyzer.analyzeText(text);
    }

    @Override
    public void setNeuralNetwork(final BasicNetwork network) {
        this.network = network;
    }

    @Override
    public boolean trainStemmedDirectory(final String directoryName) {
        return training.trainStemmedDirectory(directoryName);
    }

    @Override
    public boolean trainStemmedFile(final String fileName) {
        return training.trainStemmedFile(fileName);
    }

    @Override
    public void setCategories(Map<String, Integer> categories) {
        this.categories = categories;
    }

    @Override
    public void loadCategoriesFromFile(String fileName) {
        File categoriesFile = new File(fileName);

        try (Scanner scanner = new Scanner(categoriesFile)) {

            while (scanner.hasNextLine()) {
                String category = scanner.nextLine();
                addCategory(category);
            }
            scanner.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public BasicNetwork getNetwork() {
        return network;
    }

    @Override
    public void setMaxEpochs(final int maxEpochs) {
        training.setMaxEpochs(maxEpochs);
    }

    @Override
    public void setDesiredError(final double desiredError) {
        training.setDesiredError(desiredError);
    }

    @Override
    public void setTrainingMethodType(Class<? extends Train> trainingMethodType) {
        training.setTrainingMethodType(trainingMethodType);
    }

    @Override
    public Set<String> getTrainedWords() {
        return training.getTrainedWords();
    }

    private void addCategory(final String categoryName) {
        categories.putIfAbsent(categoryName, categories.size());
    }
}
