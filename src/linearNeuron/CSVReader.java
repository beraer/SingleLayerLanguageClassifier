package linearNeuron;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class CSVReader {

    private static final Map<String, double[]> LANG_MAP = new HashMap<>();
    static {
        LANG_MAP.put("English", new double[]{1, 0, 0, 0});
        LANG_MAP.put("Spanish", new double[]{0, 1, 0, 0});
        LANG_MAP.put("German",  new double[]{0, 0, 1, 0});
        LANG_MAP.put("Polish",  new double[]{0, 0, 0, 1});
    }

    public static List<Vector> readCSV(String csvPath) {
        List<Vector> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                int firstCommaIndex = line.indexOf(',');
                if (firstCommaIndex == -1) {
                    continue;
                }

                String language = line.substring(0, firstCommaIndex)
                        .replace("\"", "")
                        .trim();

                String text = line.substring(firstCommaIndex + 1).trim();
                text = text.replaceAll("^\"|\"$", "");

                double[] freq = SingleLayerNetwork.textToFrequencyVector(text);

                double[] target = LANG_MAP.get(language);
                if (target == null) {
                    System.err.println("Unknown language: " + language);
                    continue;
                }
                data.add(new Vector(freq, target));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }
}
