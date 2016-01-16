package backend;

import org.encog.neural.networks.BasicNetwork;

import java.util.Set;

public interface Training {
    /**
     * @return true, if OK
     */
    boolean trainStemmedDirectory(final String directoryName);

    BasicNetwork getNetwork();

    /**
     * @return all known words
     */
    Set<String> getTrainedWords();

    Set<String> getCategories();

    void addCategory(final String categoryName);
}
