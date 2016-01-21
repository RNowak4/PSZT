package backend;

import opennlp.tools.stemmer.PorterStemmer;
import org.encog.neural.data.NeuralData;
import org.encog.neural.networks.BasicNetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

/**
 * Created by wprzecho on 21.01.16.
 */
public class AnalyzerImpl implements Analyzer {
    @Override
    public NeuralData analyzeFile(String fileName) {
        final StringBuilder sb = new StringBuilder();
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));

            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                line = br.readLine();
            }
            return analyzeText(sb.toString());
        }catch (Exception e){
            e.printStackTrace();
        }
        return analyzeText("");
    }

    @Override
    public NeuralData analyzeText(String text) {
        final String[] words = text.split("\\s+|,\\s*|\\.\\s*");
        final PorterStemmer stemmer = new PorterStemmer();
        for (final String word : words) {
            final String stemmedWord =  stemmer.stem(word);
        }
        return null;
    }

    @Override
    public void setNeuralNetwork(BasicNetwork network) {

    }
}
