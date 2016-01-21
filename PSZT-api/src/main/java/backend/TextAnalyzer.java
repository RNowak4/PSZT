package backend;

import java.util.Map;

public interface TextAnalyzer extends Analyzer, Training {
    Map<String, Integer> getCategories();

    void setCategories(final Map<String, Integer> categories);

    void loadCategoriesFromFile(final String fileName);
}
