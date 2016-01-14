package backend;

import com.googlecode.fannj.ActivationFunction;
import com.googlecode.fannj.TrainingAlgorithm;

public class ControllerImpl implements Controller {
    @Override
    public void loadFannFromFile(String filePath) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void safeFannInFile(String filePath) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void setTrainingAlgorithm(TrainingAlgorithm trainingAlgorithm) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void startTraining() {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void setActivationFunction(ActivationFunction activationFunction) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void setDesiredError(double value) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void setEpochsBetweenReports(int value) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void setMaxEpochs(int value) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public void setNeuronLayersNumber(int... values) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public double[] classifyFile(String filePath) {
        throw new UnsupportedOperationException("TODO");
    }

    @Override
    public double[] classifyText(String text) {
        throw new UnsupportedOperationException("TODO");
    }
}
