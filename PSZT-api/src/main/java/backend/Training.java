package backend;

import org.encog.neural.networks.BasicNetwork;

import java.io.File;
import java.util.Set;

public interface Training {
    /**
     * @return true, if OK
     */
    boolean trainStemmedDirectory(final String directoryName);

    boolean trainStemmedFile(final String fileName);

    BasicNetwork getNetwork();

    /**
     * @return all known words
     */
    Set<String> getTrainedWords();

    Set<String> getCategories();
}
