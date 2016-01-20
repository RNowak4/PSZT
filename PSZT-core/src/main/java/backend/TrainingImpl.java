package backend;

import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SuppressWarnings("ConstantConditions")
public class TrainingImpl implements Training {
    private double desiredError;
    private int maxEpochs;
    private BasicNetwork network;
    private Class<? extends Train> trainingMethodType;
    private double[][] inputTable;
    private double[][] idealOutputTable;
    private Map<String, Integer> categories = new HashMap<>();
    private Set<String> trainedWords = new HashSet<>();

    public TrainingImpl(final double desiredError, final int maxEpochs,
                        final Class<? extends Train> trainingMethodType) {
        this.desiredError = desiredError;
        this.maxEpochs = maxEpochs;
        this.trainingMethodType = trainingMethodType;
        this.network = new BasicNetwork();

        loadDefaultCategories();
    }

    private void loadDefaultCategories() {
        ClassLoader cl = this.getClass().getClassLoader();
        File categoriesFile = new File(cl.getResource("categories.txt").getFile());

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

    private void createWordsMap(final File trainingDirectory) throws IOException {
        for (final File file : trainingDirectory.listFiles()) {
            createFileWordsMap(file);
        }
        trainedWords.removeAll(categories.keySet());
    }

    private void createFileWordsMap(final File file) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                trainedWords.addAll(Stream.of(line.split("\\s+")).collect(Collectors.toSet()));
            }
        }
    }

    private void flushTrainingData(final List<double[]> inputList, final List<double[]> idealOutputList,
                                   final Map<String, Integer> wordsMap, final int categoryPos) {
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

    private void learnFromFile(final File file,
                               final List<double[]> inputList,
                               final List<double[]> idealOutputList) throws IOException {
        final Map<String, Integer> wordsMap = new HashMap<>();

        int lastCategoryPos = -1;
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
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
        }
        flushTrainingData(inputList, idealOutputList, wordsMap, lastCategoryPos);
    }

    private void createLearningData(final File file) throws IOException {
        final List<double[]> inputList = new ArrayList<>();
        final List<double[]> idealOutputList = new ArrayList<>();

        if (file.isDirectory()) {
            for (final File subFile : file.listFiles()) {
                learnFromFile(subFile, inputList, idealOutputList);
            }
        } else
            learnFromFile(file, inputList, idealOutputList);

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
    public boolean trainStemmedFile(final String fileName) {
        final File file = new File(fileName);

        try {
            createFileWordsMap(file);
            createLearningData(file);
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
        network.addLayer(new BasicLayer(new ActivationLinear(), true, outputSize));
        network.getStructure().finalizeStructure();
        network.reset();
    }

    private void learnNetwork() {
        final NeuralDataSet trainingSet = new BasicNeuralDataSet(inputTable, idealOutputTable);
        final Train train = getTraining(network, trainingSet);

        int epoch = 0;
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error:" + train.getError());
            ++epoch;
        } while (epoch < maxEpochs && train.getError() > desiredError);

    }

    private Train getTraining(final BasicNetwork network, final NeuralDataSet trainingSet) {
        if (this.trainingMethodType.equals(ResilientPropagation.class))
            return new ResilientPropagation(network, trainingSet);
        else if (this.trainingMethodType.equals(QuickPropagation.class))
            return new QuickPropagation(network, trainingSet);
        else
            throw new UnsupportedOperationException("Unsupported Training method type!");
    }

    @Override
    public Set<String> getTrainedWords() {
        return trainedWords;
    }

    @Override
    public Set<String> getCategories() {
        return categories.keySet();
    }

    public void addCategory(final String categoryName) {
        categories.putIfAbsent(categoryName, categories.size());
    }
}
