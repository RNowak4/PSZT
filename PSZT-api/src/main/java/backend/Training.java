package backend;

import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;

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

    void setMaxEpochs(int maxEpochs);

    void setDesiredError(double desiredError);

    void setTrainingMethodType(Class<? extends Train> trainingMethodType);

    /**
     * @return all known words
     */
    Set<String> getTrainedWords();
}
