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

class DataSet {
    private int category;
    private Map<String, Integer> wordsMap;

    public DataSet(final int category) {
        this.category = category;
        this.wordsMap = new HashMap<>();
    }

    void addWord(final String word) {
        Integer count = wordsMap.get(word);
        if (count != null)
            wordsMap.replace(word, count + 1);
        else
            wordsMap.put(word, 1);
    }

    public int getCategory() {
        return category;
    }

    public Map<String, Integer> getWordsMap() {
        return wordsMap;
    }
}

@SuppressWarnings("ConstantConditions")
public class TrainingImpl implements Training {
    private double desiredError;
    private int maxEpochs;
    private BasicNetwork network;
    private Class<? extends Train> trainingMethodType;
    private double[][] inputTable;
    private double[][] idealOutputTable;
    private Map<String, Integer> categories = new HashMap<>();
    private Set<String> trainedWords = new TreeSet<>();

    public TrainingImpl(final double desiredError, final int maxEpochs,
                        final BasicNetwork network,
                        final Class<? extends Train> trainingMethodType) {
        this.desiredError = desiredError;
        this.maxEpochs = maxEpochs;
        this.trainingMethodType = trainingMethodType;
        this.network = network;
    }

    // returns list containing counter words for each category
    private List<DataSet> learnFromFile(final File file) throws IOException {
        List<DataSet> retList = new ArrayList<>();
        DataSet dataSet = null;

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                for (final String word : line.split("\\s+")) {
                    Integer categoryPos = categories.get(word);
                    if (categoryPos != null) {
                        if (dataSet != null)
                            retList.add(dataSet);
                        dataSet = new DataSet(categoryPos);
                    } else {
                        dataSet.addWord(word);
                    }
                }
            }
        }

        retList.add(dataSet);

        return retList;
    }

    private void createLearningData(final File file) throws IOException {
        // list of analyzed data for each file.
        // For each file we have list of all analyzed categories statistics in map.
        final List<List<DataSet>> learnedData = new ArrayList<>();

        if (file.isDirectory()) {
            for (final File subFile : file.listFiles()) {
                learnedData.add(learnFromFile(subFile));
            }
        } else
            learnedData.add(learnFromFile(file));

        flushData(learnedData);
    }

    private void flushData(final List<List<DataSet>> learnedData) {
        // create list of learned words
        int analyzedCategories = 0;
        for (List<DataSet> dataSets : learnedData) {
            for (DataSet dataSet : dataSets) {
                trainedWords.addAll(dataSet.getWordsMap().keySet());
            }

            analyzedCategories += dataSets.size();
        }

        trainedWords.removeAll(categories.keySet());

        inputTable = new double[analyzedCategories][];
        idealOutputTable = new double[analyzedCategories][];

        int trainedWordsNumber = trainedWords.size();
        int outputsNumber = categories.size();

        int i = 0;
        for (List<DataSet> dataSets : learnedData) {
            for (DataSet dataSet : dataSets) {
                inputTable[i] = new double[trainedWordsNumber];
                idealOutputTable[i] = new double[outputsNumber];

                int j = 0;
                for (String trainedWord : trainedWords) {
                    Integer mapValue = dataSet.getWordsMap().get(trainedWord);
                    if (mapValue == null)
                        inputTable[i][j] = 0d;
                    else
                        inputTable[i][j] = mapValue;
                    ++j;
                }

                idealOutputTable[i][dataSet.getCategory()] = 1.0d;
                ++i;
            }
        }
    }

    @Override
    public boolean trainStemmedDirectory(final String directoryName) {
        final File trainingDirectory = new File(directoryName);

        if (!trainingDirectory.isDirectory()) {
            return false;
        }

        try {
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

    @Override
    public void setMaxEpochs(int maxEpochs) {
        this.maxEpochs = maxEpochs;
    }

    @Override
    public void setDesiredError(double desiredError) {
        this.desiredError = desiredError;
    }

    @Override
    public void setTrainingMethodType(Class<? extends Train> trainingMethodType) {
        this.trainingMethodType = trainingMethodType;
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
    public void setCategories(final Map<String, Integer> categories) {
        this.categories = categories;
    }
}
