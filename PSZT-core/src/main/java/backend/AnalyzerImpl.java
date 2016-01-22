package backend;

import opennlp.tools.stemmer.PorterStemmer;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Created by wprzecho
 */
public class AnalyzerImpl implements Analyzer {

    private Training training;

    public AnalyzerImpl(final Training training) {
        this.training = training;
    }

    @Override
    public MLData analyzeFile(final String fileName) {
        final StringBuilder sb = new StringBuilder();
        try {
            final BufferedReader br = new BufferedReader(new FileReader(fileName));

            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                line = br.readLine();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return analyzeText(sb.toString());
    }

    @Override
    public MLData analyzeText(final String text) {
        final Map<String, Integer> tempResult = new HashMap<>();
        final String[] words = text.split("\\s+|,\\s*|\\.\\s*");
        final PorterStemmer stemmer = new PorterStemmer();
        final Set<String> stopWords = training.getStopWords();
        for (final String word : words) {
            if(stopWords.contains(word))
                continue;

            final String stemmedWord = stemmer.stem(word);
            Integer amount = tempResult.get(stemmedWord);
            if (amount == null)
                tempResult.put(stemmedWord, 1);
            else {
                tempResult.put(stemmedWord, ++amount);
            }
        }
        final String[] allWords = training.getTrainedWords().toArray(new String[training.getTrainedWords().size()]);
        double[] wordsTable = new double[allWords.length];
        for (int i = 0; i < allWords.length; ++i) {
            if (tempResult.containsKey(allWords[i])) {
                wordsTable[i] = tempResult.get(allWords[i]);
            } else
                wordsTable[i] = 0;
        }

        final BasicNetwork network = training.getNetwork();
        return network.compute(new BasicMLData(wordsTable));
    }
}
