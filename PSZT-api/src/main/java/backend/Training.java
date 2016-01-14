package backend;

import com.googlecode.fannj.Fann;
import com.googlecode.fannj.TrainingAlgorithm;

import java.io.File;
import java.util.Optional;
import java.util.Set;

public interface Training {
    Optional<Fann> trainStemmedDirectory();

    Optional<Fann> getFann();

    Set<String> getTrainedWords();

    Set<String> getCategories();

    void addCategory(String categoryName);
}
