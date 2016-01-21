package backend;

import java.util.Map;

public interface TextAnalyzer extends Analyzer, Training {
    Map<String, Integer> getCategories();

    void loadCategoriesFromFile(final String fileName);
}
