package backend;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SuppressWarnings("ConstantConditions")
public class TrainingImpl implements Training {
    private double desiredError;
    private int maxEpochs;
    private BasicNetwork network;
    private double[][] inputTable;
    private double[][] idealOutputTable;
    private Map<String, Integer> categories = new HashMap<>();
    private Set<String> trainedWords = new TreeSet<>(); // sorted

    public TrainingImpl(final double desiredError, final int maxEpochs) {
        this.desiredError = desiredError;
        this.maxEpochs = maxEpochs;
        this.network = new BasicNetwork();
    }

    private void createWordsMap(final File trainingDirectory) throws IOException {
        for (final File file : trainingDirectory.listFiles()) {
            Path path = FileSystems.getDefault().getPath(trainingDirectory.getName(), file.getName());
            for (String line : Files.readAllLines(path)) {
                trainedWords.addAll(Stream.of(line.split("\\s+")).collect(Collectors.toSet()));
            }
        }
        trainedWords.removeAll(categories.keySet());
    }

    private void flushTrainingData(List<double[]> inputList, List<double[]> idealOutputList, final Map<String, Integer> wordsMap, final int categoryPos) {
        boolean wrongArguments = wordsMap.isEmpty() || categoryPos < 0;
        if (wrongArguments)
            return;

        // filling input table
        int inputSize = trainedWords.size();
        final double[] inputToAdd = new double[inputSize];
        inputList.add(inputToAdd);

        int i = 0;
        for (String trainingWord : trainedWords) {
            if (wordsMap.containsKey(trainingWord))
                inputToAdd[i] = (wordsMap.get(trainingWord));
            else
                inputToAdd[i] = 0d;
            ++i;
        }

        // filling ideal output table
        int outputSize = categories.size();
        final double[] idealOutput = new double[outputSize];
        idealOutputList.add(idealOutput);
        idealOutput[categoryPos] = 1d;

        wordsMap.clear();
    }

    private void learnFromFile(final Path filePath, List<double[]> inputList, List<double[]> idealOutputList) throws IOException {
        final Map<String, Integer> wordsMap = new HashMap<>();

        int lastCategoryPos = -1;
        for (final String line : Files.readAllLines(filePath)) {
            for (final String word : line.split("\\s+")) {
                if (categories.containsKey(word)) {
                    flushTrainingData(inputList, idealOutputList, wordsMap, lastCategoryPos);
                    lastCategoryPos = categories.get(word);
                } else {
                    if (wordsMap.containsKey(word))
                        wordsMap.replace(word, wordsMap.get(word) + 1);
                    else
                        wordsMap.put(word, 1);
                }
            }
        }
        flushTrainingData(inputList, idealOutputList, wordsMap, lastCategoryPos);
    }

    private void createLearningData(final File trainingDirectory) throws IOException {
        final List<double[]> inputList = new ArrayList<>();
        final List<double[]> idealOutputList = new ArrayList<>();

        for (final File file : trainingDirectory.listFiles()) {
            Path filePath = FileSystems.getDefault().getPath(file.getParent(), file.getName());
            learnFromFile(filePath, inputList, idealOutputList);
        }

        int inputListSize = inputList.size();
        int outputListSize = idealOutputList.size();

        inputTable = new double[inputListSize][];
        idealOutputTable = new double[outputListSize][];

        inputList.toArray(inputTable);
        idealOutputList.toArray(idealOutputTable);
    }

    @Override
    public boolean trainStemmedDirectory(final String directoryName) {
        final File trainingDirectory = new File(directoryName);
        if (!trainingDirectory.isDirectory()) {
            return false;
        }

        try {
            createWordsMap(trainingDirectory);
            createLearningData(trainingDirectory);
        } catch (IOException e) {
            e.printStackTrace();
        }

        prepareNetwork();
        learnNetwork();
        return true;
    }

    @Override
    public BasicNetwork getNetwork() {
        return network;
    }

    private void prepareNetwork() {
        int inputSize = this.trainedWords.size();
        int outputSize = this.categories.size();

        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, inputSize));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 2 * inputSize));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, outputSize));
        network.getStructure().finalizeStructure();
        network.reset();
    }

    private void learnNetwork() {
        final NeuralDataSet trainingSet = new BasicNeuralDataSet(inputTable, idealOutputTable);
        final Train train = new ResilientPropagation(network, trainingSet);

        int epoch = 0;
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            ++epoch;
        } while (epoch <= maxEpochs && train.getError() > desiredError);

    }

    @Override
    public Set<String> getTrainedWords() {
        return trainedWords;
    }

    @Override
    public Set<String> getCategories() {
        return categories.keySet();
    }

    @Override
    public void addCategory(final String categoryName) {
        categories.putIfAbsent(categoryName, categories.size());
    }
}