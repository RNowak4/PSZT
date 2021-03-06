package backend;

import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TextAnalyzerImpl implements TextAnalyzer {
    private Analyzer analyzer;
    private BasicNetwork network;
    private Training training;
    private Map<String, Integer> categories;
    private Set<String> stopWords = new HashSet<>();

    public TextAnalyzerImpl(double desiredError, int maxEpochs,
                            final Class<? extends Train> trainingMethodType) {
        this.categories = new HashMap<>();
        this.network = new BasicNetwork();
        loadStopWords();
        this.training = new TrainingImpl(desiredError, maxEpochs, network, stopWords, trainingMethodType);
        this.training.setCategories(categories);
        this.analyzer = new AnalyzerImpl(training);
    }

    @Override
    public Map<String, Integer> getCategories() {
        return categories;
    }

    @Override
    public MLData analyzeFile(final String fileName) {
        return analyzer.analyzeFile(fileName);
    }

    @Override
    public MLData analyzeText(final String text) {
        return analyzer.analyzeText(text);
    }

    @Override
    public void setNeuralNetwork(final BasicNetwork network) {
        this.network = network;
    }

    @Override
    public boolean trainDirectory(final String directoryName) {
        return training.trainDirectory(directoryName);
    }

    @Override
    public boolean trainFile(final String fileName) {
        return training.trainFile(fileName);
    }

    @Override
    public boolean trainFileWithStatistics(String fileName) {
        return training.trainFileWithStatistics(fileName);
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

    private void loadStopWords() {
        ClassLoader cl = this.getClass().getClassLoader();
        File stopWordsFile = new File(cl.getResource("stopWords.txt").getFile());

        try (BufferedReader br = new BufferedReader(new FileReader(stopWordsFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                stopWords.addAll(Stream.of(line.split("\\s+")).collect(Collectors.toSet()));
            }
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
    public Set<String> getStopWords() {
        return training.getStopWords();
    }

    @Override
    public Set<String> getTrainedWords() {
        return training.getTrainedWords();
    }

    @Override
    public boolean isWordKnown(String word) {
        return training.getTrainedWords().contains(word);
    }

    private void addCategory(final String categoryName) {
        int size = categories.size();
        categories.putIfAbsent(categoryName, size);
    }
}
