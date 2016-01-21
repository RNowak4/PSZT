package backend;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by radek on 1/21/16.
 */
class DataSet {
    private int category;
    private Map<String, Integer> wordsMap;

    public DataSet(final int category) {
        this.category = category;
        this.wordsMap = new HashMap<>();
    }

    void addWord(final String word) {
        Integer count = wordsMap.get(word);
        if (count != null)
            wordsMap.replace(word, count + 1);
        else
            wordsMap.put(word, 1);
    }

    public int getCategory() {
        return category;
    }

    public Map<String, Integer> getWordsMap() {
        return wordsMap;
    }
}
