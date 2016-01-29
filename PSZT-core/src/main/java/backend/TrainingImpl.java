package backend;

import opennlp.tools.stemmer.PorterStemmer;
import opennlp.tools.stemmer.Stemmer;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
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
    private Set<String> stopWords;
    private List<DataSet> allDataSets = new ArrayList<>();

    public TrainingImpl(final double desiredError, final int maxEpochs,
                        final BasicNetwork network,
                        final Set<String> stopWords,
                        final Class<? extends Train> trainingMethodType) {
        this.desiredError = desiredError;
        this.maxEpochs = maxEpochs;
        this.trainingMethodType = trainingMethodType;
        this.network = network;
        this.stopWords = stopWords;
    }

    // returns list containing counter words for each category
    private List<DataSet> learnFromFile(final File file) throws IOException {
        final List<DataSet> retList = new ArrayList<>();
        final Stemmer stemmer = new PorterStemmer();
        DataSet dataSet = null;

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                final String[] wordsInLine = line.split("\\s+");
                for (final String word : wordsInLine) {
                    if (stopWords.contains(word))
                        continue;

                    final String stemmedWord = stemmer.stem(word).toString();
                    final Integer categoryPos = categories.get(word);
                    if (categoryPos != null) {
                        if (dataSet != null)
                            retList.add(dataSet);
                        dataSet = new DataSet(categoryPos);
                    } else {
                        dataSet.addWord(stemmedWord);
                    }
                }
            }
        }
        retList.add(dataSet);

        return retList;
    }

    @Override
    public Set<String> getStopWords() {
        return stopWords;
    }

    private void createLearningData(final File file) throws IOException {
        // list of analyzed data for each file.
        // For each file we have list of all analyzed categories statistics in map.
        final List<List<DataSet>> learnedData = new ArrayList<>();

        if (file.isDirectory()) {
            for (final File subFile : file.listFiles()) {
                final List<DataSet> dataSets = learnFromFile(subFile);
                learnedData.add(dataSets);
                allDataSets.addAll(dataSets);
            }
        } else {
            final List<DataSet> dataSets = learnFromFile(file);
            learnedData.add(dataSets);
            allDataSets.addAll(dataSets);
        }

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
    public boolean trainDirectory(final String directoryName) {
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
        teachNetwork();

        return true;
    }

    @Override
    public boolean trainFile(final String fileName) {
        final File file = new File(fileName);

        try {
            createLearningData(file);
        } catch (IOException e) {
            e.printStackTrace();
        }

        prepareNetwork();
        teachNetwork();

        return true;
    }

    @Override
    public boolean trainFileWithStatistics(final String fileName) {
        final File file = new File(fileName);

        try {
            createLearningData(file);
        } catch (IOException e) {
            e.printStackTrace();
        }

        prepareNetwork();
        printStatistics();

        return true;
    }

    /**
     * Funkcja powinna tworzyc tabelke z Recall i Precision.
     * Interesuje nas tylko przeczytanie danych wiec niech wywali wyniki na System.out
     */
    private void printStatistics() {
        final int allSets = allDataSets.size();
        int downSeparator = 0;
        int upSeparator = allSets / 10;

        for (int i = 0; i < 10; i++) {
            // bierzemy dane do uczenia
            final List<DataSet> learningSet = allDataSets.subList(0, downSeparator - 1);
            learningSet.addAll(allDataSets.subList(upSeparator + 1, allSets - 1));

            final List<DataSet> trainingSet = allDataSets.subList(downSeparator, upSeparator);

            // TODO uczenie sieci tutaj

            final int[] recallTable = new int[categories.size()];
            final int[] precisionTable = new int[categories.size()];
            for (DataSet dataSet : trainingSet) {
                final Integer classifiedCategory = classifyDataSet(dataSet);
                recallTable[classifiedCategory]++;

                if (classifiedCategory == dataSet.getCategory())
                    precisionTable[classifiedCategory]++;
            }

            // dla kazdej kategorii wyswietlamy recall a na koniec wyswietlamy precision(1 kolumna)
            System.out.println("Recall for iteration " + i + ":");
            int j = 0;
            for (int count : recallTable) {
                // liczenie recalla dla kazdej kategorii
                final double recall = (double) count / allDataSets.size();
                System.out.print(j++ + recall + "   ");
            }
            System.out.println();

            // liczenie precyzji
            System.out.println("Precision for iteration " + i + ":");
            j = 0;
            for (int count : precisionTable) {
                final double precision = (double) count / (double) recallTable[j];
                System.out.println(j++ + precision + "   ");
            }
            System.out.println();

            downSeparator = upSeparator;
            upSeparator += allSets / 10;
        }
    }

    private Integer classifyDataSet(final DataSet dataSet) {
        final Map<String, Integer> wordsMap = dataSet.getWordsMap();
        final String[] allKnownWords = new String[trainedWords.size()];
        final double[] wordsTable = new double[trainedWords.size()];
        trainedWords.toArray(allKnownWords);

        for (int i = 0; i < allKnownWords.length; ++i) {
            if (wordsMap.containsKey(allKnownWords[i])) {
                wordsTable[i] = wordsMap.get(allKnownWords[i]);
            } else
                wordsTable[i] = 0;
        }

        MLData computed = network.compute(new BasicMLData(wordsTable));

        double max = 0.0;
        int classifiedCategory = 0;
        int counter = 0;
        for (double v : computed.getData()) {
            if (v > max) {
                max = v;
                classifiedCategory = counter;
            }
            ++counter;
        }

        return classifiedCategory;
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
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, outputSize));
        network.getStructure().finalizeStructure();
        network.reset();
    }

    private void teachNetwork() {
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
    public boolean isWordKnown(String word) {
        return trainedWords.contains(word);
    }

    @Override
    public void setCategories(final Map<String, Integer> categories) {
        this.categories = categories;
    }
}
