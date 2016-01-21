package backend;

import org.encog.neural.networks.BasicNetwork;

import java.util.Map;
import java.util.Set;

public interface Training {
    /**
     * @return true, if OK
     */
    boolean trainStemmedDirectory(final String directoryName);

    boolean trainStemmedFile(final String fileName);

    void setCategories(Map<String, Integer> categories);

    BasicNetwork getNetwork();

    /**
     * @return all known words
     */
    Set<String> getTrainedWords();
}
