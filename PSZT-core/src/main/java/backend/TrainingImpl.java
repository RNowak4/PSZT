package backend;

import com.googlecode.fannj.Fann;
import com.googlecode.fannj.TrainingAlgorithm;

import java.io.*;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SuppressWarnings("ConstantConditions")
public class TrainingImpl implements Training {
    private static final String OUTPUT_FILE_NAME = "output.txt";
    private double desiredError;
    private int trainingSetsCounter;
    private int epochsBetweenReports;
    private int maxEpochs;
    private Fann fann;
    private File trainingDirectory;
    private File outputFile = new File(OUTPUT_FILE_NAME);
    private TrainingAlgorithm trainingAlgorithm;
    private Map<String, Integer> categories = new HashMap<>();
    private Set<String> trainedWords = new TreeSet<>(); // sorted

    public TrainingImpl(double desiredError, int epochsBetweenReports, int maxEpochs, String directoryName, TrainingAlgorithm trainingAlgorithm) {
        this.desiredError = desiredError;
        this.epochsBetweenReports = epochsBetweenReports;
        this.maxEpochs = maxEpochs;
        this.trainingAlgorithm = trainingAlgorithm;
        this.trainingDirectory = new File(directoryName);
    }

    private void createWordsMap() throws IOException {
        for (File file : trainingDirectory.listFiles()) {
            Path path = FileSystems.getDefault().getPath(trainingDirectory.getName(), file.getName());
            for (String line : Files.readAllLines(path)) {
                trainedWords.addAll(Stream.of(line.split("\\s+")).collect(Collectors.toSet()));
            }
        }
        trainedWords.removeAll(categories.keySet());
    }

    private void appendTrainingFile(int categoryPos, Map<String, Integer> wordsMap, Writer writer) throws IOException {
        if (wordsMap.isEmpty() || categoryPos < 0)
            return;

        StringBuilder builder = new StringBuilder();
        for (String trainingWord : trainedWords) {
            if (wordsMap.containsKey(trainingWord))
                builder.append(wordsMap.get(trainingWord));
            else
                builder.append("0");
            builder.append(" ");
        }
        builder.append("\n");

        for (int i = 0; i < categories.size(); i++) {
            if (categoryPos == i)
                builder.append("1");
            else
                builder.append("0");
            builder.append(" ");
        }
        builder.append("\n");
        writer.write(builder.toString());
    }

    private void createLearningData(File file) throws IOException {
        Path path = FileSystems.getDefault().getPath(file.getParent(), file.getName());
        Map<String, Integer> wordsMap = new HashMap<>();

        int lastCategoryPos = -1;
        Writer writer = new PrintWriter(outputFile);
        try {
            fillWithWhiteSpaces(writer);
            for (String line : Files.readAllLines(path)) {
                for (String word : line.split("\\s+")) {
                    if (categories.containsKey(word)) {
                        ++trainingSetsCounter;
                        appendTrainingFile(lastCategoryPos, wordsMap, writer);
                        lastCategoryPos = categories.get(word);
                        wordsMap.clear();
                    } else {
                        if (wordsMap.containsKey(word))
                            wordsMap.replace(word, wordsMap.get(word) + 1);
                        else
                            wordsMap.put(word, 1);
                    }
                }
            }
            appendTrainingFile(lastCategoryPos, wordsMap, writer);
        } catch (IOException e) {
            throw e;
        } finally {
            writer.close();
        }
    }

    private void fillWithWhiteSpaces(Writer writer) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 64; ++i) {
            sb.append(" ");
        }
        sb.append("\n");
        writer.write(sb.toString());
    }

    @Override
    public Optional<Fann> trainStemmedDirectory() {
        if (!trainingDirectory.isDirectory()) {
            return Optional.empty();
        }

        try {
            createWordsMap();
            for (File file : trainingDirectory.listFiles()) {
                createLearningData(file);
            }
            writeFileBeginning();
        } catch (IOException e) {
            return Optional.empty();
        }
        // TODO
        // tworzenie sieci
        // trenowanie sieci za pomoca trainingString
//        return Optional.of(fann);
        return Optional.empty();
    }

    @Override
    public Optional<Fann> getFann() {
        if (fann == null)
            return Optional.empty();
        return Optional.of(fann);
    }

    private void writeFileBeginning() throws IOException {
        RandomAccessFile randomAccessFile = null;
        try {
            randomAccessFile = new RandomAccessFile(outputFile, "rw");
            randomAccessFile.seek(0);
            randomAccessFile.writeChars(trainingSetsCounter + " " + trainedWords.size() + " " + categories.size());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            randomAccessFile.close();
        }
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
    public void addCategory(String categoryName) {
        categories.putIfAbsent(categoryName, categories.size());
    }
}
