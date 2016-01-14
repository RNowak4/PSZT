package backend;

import com.googlecode.fannj.ActivationFunction;
import com.googlecode.fannj.TrainingAlgorithm;

public interface Controller {
    void loadFannFromFile(String filePath);

    void safeFannInFile(String filePath);

    void setTrainingAlgorithm(TrainingAlgorithm trainingAlgorithm);

    void startTraining();

    void setActivationFunction(ActivationFunction activationFunction);

    void setDesiredError(double value);

    void setEpochsBetweenReports(int value);

    void setMaxEpochs(int value);

    /**
     * Sets number of neurons for each layer
     * First input neurons, Last output neurons
     */
    void setNeuronLayersNumber(int... values);

    /**
     * @return % of chance to belong to each category
     */
    double[] classifyFile(String filePath);

    double[] classifyText(String text);

//    void setView(View view);
}
