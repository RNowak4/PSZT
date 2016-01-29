package backend;

import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;

import java.util.Map;
import java.util.Set;

public interface Training {
    /**
     * @return true, if OK
     */
    boolean trainDirectory(final String directoryName);

    boolean trainFile(final String fileName);

    boolean trainFileWithStatistics(final String fileName);

    void setCategories(Map<String, Integer> categories);

    BasicNetwork getNetwork();

    void setMaxEpochs(int maxEpochs);

    void setDesiredError(double desiredError);

    void setTrainingMethodType(Class<? extends Train> trainingMethodType);

    Set<String> getStopWords();

    /**
     * @return all known words
     */
    Set<String> getTrainedWords();

    boolean isWordKnown(final String word);
}
